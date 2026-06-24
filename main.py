"""
Threadwell Smartlead Bridge
===========================
A small, guard-railed HTTP service that sits between the Threadwell managed agent
and the Smartlead API. The agent never gets the raw Smartlead key or raw send
power; it can only call the safe, capped endpoints exposed here.

Safety model
------------
- The Smartlead API key lives ONLY in this service (Railway env var SMARTLEAD_API_KEY).
- Every endpoint (except /health) requires a bearer token (BRIDGE_TOKEN) so only
  the agent can call the bridge.
- Loading leads is capped per call and per day.
- This bridge intentionally does NOT expose campaign activation / sending.
  Starting a campaign (the actual send) stays a manual, human step in the
  Smartlead UI -- this structurally enforces "Smartlead activation always
  requires Slack approval".

Two ways to call it (same safety model, same caps):
1. REST endpoints (original)        -> bash/curl with Authorization: Bearer <BRIDGE_TOKEN>
2. MCP endpoint  at /agent/mcp      -> the managed agent connects as a remote MCP
   server (streamable-HTTP). The MCP tools reuse the exact same Smartlead helper
   and the exact same per-call / per-day caps, and are bearer-guarded with the
   same BRIDGE_TOKEN. There is still NO send / activation tool.

Endpoints
---------
GET  /health                  -> liveness check (no auth)
GET  /status                  -> bridge config + today's usage (auth)
GET  /campaigns               -> list Smartlead campaigns (read-only, auth)
GET  /campaigns/{id}/leads    -> list leads in a campaign (read-only, auth)
POST /load-drafts             -> add up to MAX_LEADS_PER_CALL leads to a
                                 campaign as staged drafts; does NOT send (auth)
MCP  /agent/mcp               -> same capabilities as MCP tools (auth)
"""

import os
import contextlib
import datetime as dt
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, PlainTextResponse

SMARTLEAD_BASE = "https://server.smartlead.ai/api/v1"
SMARTLEAD_API_KEY = os.environ.get("SMARTLEAD_API_KEY", "")
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "")

# Caps (override via env if needed)
MAX_LEADS_PER_CALL = int(os.environ.get("MAX_LEADS_PER_CALL", "25"))
MAX_LEADS_PER_DAY = int(os.environ.get("MAX_LEADS_PER_DAY", "200"))

# Very simple in-memory daily counter. Resets on restart; fine for an MVP guard.
_usage: dict[str, int] = {}


def _today() -> str:
    return dt.date.today().isoformat()


def _used_today() -> int:
    return _usage.get(_today(), 0)


def _add_usage(n: int) -> None:
    _usage[_today()] = _used_today() + n


def _check_auth(authorization: str | None) -> None:
    if not BRIDGE_TOKEN:
        raise HTTPException(500, "Bridge not configured: BRIDGE_TOKEN is missing.")
    expected = f"Bearer {BRIDGE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing bridge token.")


def _require_key() -> None:
    if not SMARTLEAD_API_KEY:
        raise HTTPException(500, "Bridge not configured: SMARTLEAD_API_KEY is missing.")


async def _smartlead(method: str, path: str, *, params: dict | None = None,
                     json: Any = None) -> Any:
    _require_key()
    params = dict(params or {})
    params["api_key"] = SMARTLEAD_API_KEY
    url = f"{SMARTLEAD_BASE}{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(method, url, params=params, json=json)
    if resp.status_code >= 400:
        # Never echo the api_key back in errors
        raise HTTPException(resp.status_code, f"Smartlead API error: {resp.text[:500]}")
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text[:500]}


class Lead(BaseModel):
    email: str
    first_name: str | None = None
    last_name: str | None = None
    company_name: str | None = None
    custom_fields: dict[str, Any] | None = None


class LoadDraftsRequest(BaseModel):
    campaign_id: int = Field(..., description="Smartlead campaign id to stage leads into")
    leads: list[Lead] = Field(..., description="Leads to stage (NOT sent)")


# ---------------------------------------------------------------------------
# Shared core logic (used by BOTH the REST routes and the MCP tools)
# ---------------------------------------------------------------------------

def _enforce_caps(n: int) -> None:
    if n == 0:
        raise HTTPException(400, "No leads provided.")
    if n > MAX_LEADS_PER_CALL:
        raise HTTPException(
            400,
            f"Per-call cap exceeded: {n} leads requested, max {MAX_LEADS_PER_CALL} per call.",
        )
    if _used_today() + n > MAX_LEADS_PER_DAY:
        raise HTTPException(
            429,
            f"Daily cap would be exceeded: {_used_today()} already loaded today, "
            f"{n} requested, daily max {MAX_LEADS_PER_DAY}.",
        )


async def _load_drafts_core(campaign_id: int, leads: list[dict]) -> dict:
    n = len(leads)
    _enforce_caps(n)
    result = await _smartlead(
        "POST",
        f"/campaigns/{campaign_id}/leads",
        json={"lead_list": leads},
    )
    _add_usage(n)
    return {
        "ok": True,
        "campaign_id": campaign_id,
        "leads_staged": n,
        "leads_loaded_today": _used_today(),
        "sent": False,
        "note": "Leads staged as drafts only. They will NOT send until the campaign is started manually in Smartlead.",
        "smartlead_response": result,
    }


def _status_payload() -> dict:
    return {
        "ok": True,
        "date": _today(),
        "leads_loaded_today": _used_today(),
        "max_leads_per_call": MAX_LEADS_PER_CALL,
        "max_leads_per_day": MAX_LEADS_PER_DAY,
        "sending_via_bridge": False,
        "note": "Activation/sending is intentionally not exposed. Start campaigns manually in Smartlead after Slack approval.",
    }


# ---------------------------------------------------------------------------
# MCP server (streamable-HTTP). Mounted at /agent -> endpoint is /agent/mcp.
# Built BEFORE the FastAPI app so the session manager exists for the lifespan.
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "threadwell-smartlead",
    # The managed-agent proxy connects over the public Railway host. FastMCP's default
    # DNS-rebinding protection only allows localhost Host headers and would reject that
    # connection with 421. The /agent/mcp endpoint is bearer-guarded (below) and sits
    # behind Railway's HTTPS proxy, so the localhost host check is unnecessary here.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
async def smartlead_status() -> dict:
    """Return the bridge's config and today's lead-loading usage (read-only).
    Use this to confirm the bridge is reachable and how many leads have been staged today."""
    return _status_payload()


@mcp.tool()
async def smartlead_list_campaigns() -> Any:
    """List all Smartlead campaigns (read-only). Returns campaign ids, names, and status.
    Use this to find the campaign id to stage an approved batch into."""
    return await _smartlead("GET", "/campaigns")


@mcp.tool()
async def smartlead_list_campaign_leads(campaign_id: int) -> Any:
    """List the leads currently in a Smartlead campaign (read-only).
    Use this to verify what is already staged and avoid double-loading."""
    return await _smartlead("GET", f"/campaigns/{campaign_id}/leads")


@mcp.tool()
async def smartlead_load_drafts(campaign_id: int, leads: list[Lead]) -> dict:
    """Stage (NOT send) up to the per-call cap of leads into a Smartlead campaign as drafts.

    This is the auto-stage step for an APPROVED batch. Each lead: email (required),
    first_name, last_name, company_name, custom_fields (dict, e.g. custom_subject /
    custom_body for the {{custom_subject}} / {{custom_body}} sequence variables).

    IMPORTANT: This only stages drafts. It does NOT send. Sending still requires a
    human to start the campaign in the Smartlead UI -- there is no send/activation tool."""
    lead_dicts = [l.model_dump(exclude_none=True) for l in leads]
    return await _load_drafts_core(campaign_id, lead_dicts)


# Build the streamable-HTTP ASGI app and bearer-guard it.
mcp_asgi = mcp.streamable_http_app()


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.rstrip("/").endswith("/health"):
            return PlainTextResponse("ok")
        if not BRIDGE_TOKEN:
            return JSONResponse({"error": "Bridge not configured: BRIDGE_TOKEN missing."}, status_code=500)
        if request.headers.get("authorization", "") != f"Bearer {BRIDGE_TOKEN}":
            return JSONResponse({"error": "Invalid or missing bridge token."}, status_code=401)
        return await call_next(request)


mcp_asgi.add_middleware(BearerAuthMiddleware)


@contextlib.asynccontextmanager
async def lifespan(_app: "FastAPI"):
    # A mounted sub-app's own lifespan is not run by the parent, so we start the
    # MCP session manager here for the life of the process.
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="Threadwell Smartlead Bridge", version="1.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# REST endpoints (original behaviour, unchanged)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "threadwell-smartlead-bridge"}


@app.get("/status")
async def status_endpoint(authorization: str | None = Header(default=None)) -> dict:
    _check_auth(authorization)
    return _status_payload()


@app.get("/campaigns")
async def list_campaigns(authorization: str | None = Header(default=None)) -> Any:
    _check_auth(authorization)
    return await _smartlead("GET", "/campaigns")


@app.get("/campaigns/{campaign_id}/leads")
async def list_campaign_leads(campaign_id: int,
                              authorization: str | None = Header(default=None)) -> Any:
    _check_auth(authorization)
    return await _smartlead("GET", f"/campaigns/{campaign_id}/leads")


@app.post("/load-drafts")
async def load_drafts(req: LoadDraftsRequest,
                      authorization: str | None = Header(default=None)) -> dict:
    _check_auth(authorization)
    lead_list = [l.model_dump(exclude_none=True) for l in req.leads]
    return await _load_drafts_core(req.campaign_id, lead_list)


# Mount the MCP app last. Endpoint: https://<host>/agent/mcp
app.mount("/agent", mcp_asgi)
