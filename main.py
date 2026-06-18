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

Endpoints
---------
GET  /health                      -> liveness check (no auth)
GET  /status                      -> bridge config + today's usage (auth)
GET  /campaigns                   -> list Smartlead campaigns (read-only, auth)
GET  /campaigns/{id}/leads        -> list leads in a campaign (read-only, auth)
POST /load-drafts                 -> add up to MAX_LEADS_PER_CALL leads to a
                                     campaign as staged drafts; does NOT send (auth)
"""

import os
import datetime as dt
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

SMARTLEAD_BASE = "https://server.smartlead.ai/api/v1"
SMARTLEAD_API_KEY = os.environ.get("SMARTLEAD_API_KEY", "")
BRIDGE_TOKEN = os.environ.get("BRIDGE_TOKEN", "")

# Caps (override via env if needed)
MAX_LEADS_PER_CALL = int(os.environ.get("MAX_LEADS_PER_CALL", "25"))
MAX_LEADS_PER_DAY = int(os.environ.get("MAX_LEADS_PER_DAY", "200"))

app = FastAPI(title="Threadwell Smartlead Bridge", version="1.0.0")

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


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "threadwell-smartlead-bridge"}


@app.get("/status")
async def status_endpoint(authorization: str | None = Header(default=None)) -> dict:
    _check_auth(authorization)
    return {
        "ok": True,
        "date": _today(),
        "leads_loaded_today": _used_today(),
        "max_leads_per_call": MAX_LEADS_PER_CALL,
        "max_leads_per_day": MAX_LEADS_PER_DAY,
        "sending_via_bridge": False,
        "note": "Activation/sending is intentionally not exposed. Start campaigns manually in Smartlead after Slack approval.",
    }


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

    n = len(req.leads)
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

    lead_list = [l.model_dump(exclude_none=True) for l in req.leads]
    result = await _smartlead(
        "POST",
        f"/campaigns/{req.campaign_id}/leads",
        json={"lead_list": lead_list},
    )
    _add_usage(n)
    return {
        "ok": True,
        "campaign_id": req.campaign_id,
        "leads_staged": n,
        "leads_loaded_today": _used_today(),
        "sent": False,
        "note": "Leads staged as drafts only. They will NOT send until the campaign is started manually in Smartlead.",
        "smartlead_response": result,
    }
