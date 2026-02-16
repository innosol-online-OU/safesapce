# Cloudflare Setup Guide - Split Architecture

This guide details how to configure Cloudflare for the new split architecture (Frontend + Backend on separate or same servers).

## Architecture Overview
- **Frontend (UI)**: React Static App (served via Nginx).
- **Backend (GPU)**: FastAPI Server (Python/Torch).
- **Goal**: Serve UI on `safespace.innosol.online` and API on `api.safespace.innosol.online` (or via Tunnel).

## 1. Domain & DNS Configuration

### A. Frontend (UI Server)
1.  Go to **Cloudflare Dashboard** > **DNS**.
2.  Add an **A Record**:
    -   **Name**: `@` (or `safespace` subdomain)
    -   **IPv4 Address**: `[IP_OF_UI_SERVER]`
    -   **Proxy Status**: Proxied (Orange Cloud)

### B. Backend (GPU Server)
Since the GPU server might be behind a firewall or NAT, we recommend using **Cloudflare Tunnels**.

#### Option 1: Cloudflare Tunnel (Recommended)
1.  Install `cloudflared` on the GPU Server.
2.  Login: `cloudflared tunnel login`.
3.  Create Tunnel: `cloudflared tunnel create safespace-gpu`.
4.  Configure `config.yml`:
    ```yaml
    tunnel: [UUID]
    credentials-file: /root/.cloudflared/[UUID].json
    
    ingress:
      - hostname: api.safespace.innosol.online
        service: http://localhost:8000
      - service: http_status:404
    ```
5.  Route DNS: `cloudflared tunnel route dns safespace-gpu api.safespace.innosol.online`.
6.  Run Tunnel: `cloudflared tunnel run safespace-gpu`.

#### Option 2: Direct Port Exposure (Not Recommended)
1.  Add **A Record**:
    -   **Name**: `api`
    -   **IPv4 Address**: `[IP_OF_GPU_SERVER]`
    -   **Proxy Status**: Proxied.
2.  Ensure Port 8000 is open on the server firewall.

## 2. Page Rules & Caching

1.  Go to **Rules** > **Page Rules**.
2.  **Rule 1: Bypass Cache for API**
    -   **URL**: `api.safespace.innosol.online/*` (or `safespace.innosol.online/api/*`)
    -   **Setting**: `Cache Level` = **Bypass**.
    -   **Setting**: `SSL` = **Full (Strict)**.

3.  **Rule 2: Cache Frontend**
    -   **URL**: `safespace.innosol.online/*`
    -   **Setting**: `Cache Level` = **Cache Everything**.
    -   **Setting**: `Edge Cache TTL` = **2 hours**.

## 3. Environment Variables
Ensure the Frontend Docker container (or build process) has the correct API URL.

- **Variable**: `VITE_API_URL`
- **Value**: `https://api.safespace.innosol.online` (If using subdomain)
- **Value**: `/api` (If using single domain + Nginx proxy)
