# Web Development Guide

## Issue Resolution Summary

### Problem
The `formatGeneratedResponse` function in the chat interface was not being updated in the browser despite making changes to `index.html` and restarting Docker containers with `docker compose down` and `docker compose up -d`.

### Root Cause
The web service in Docker Compose was building a static image that copied the HTML file at build time. Simply restarting containers doesn't rebuild the image, so changes weren't reflected.

### Solutions Implemented

#### 1. Immediate Fix (Force Rebuild)
```bash
docker compose down
docker compose build web
docker compose up -d
```

#### 2. Development-Friendly Setup (Volume Mounting)
Modified `docker-compose.yml` to add volume mounting:
```yaml
web:
  build: ./web
  ports:
    - "3000:3000"
  volumes:
    - ./web/index.html:/usr/share/nginx/html/index.html  # 🆕 Added this line
  environment:
    - REACT_APP_API_URL=http://localhost:8000
  depends_on:
    - app
  networks:
    - knowledge-graph-net
```

#### 3. Development Scripts
Created helper scripts for easier development:

- `dev.sh` - Main development script with commands:
  - `./dev.sh start-all` - Start all services (with rebuild)
  - `./dev.sh start-backend` - Start only backend services
  - `./dev.sh start-web` - Start web dev server with live reloading
  - `./dev.sh rebuild-web` - Force rebuild web service only
  - `./dev.sh stop` - Stop all services
  - `./dev.sh logs` - Show logs

- `web/dev-server.py` - Python development server with live reloading

## Development Workflows

### For Active Frontend Development
1. Start backend services only:
   ```bash
   ./dev.sh start-backend
   ```

2. In a separate terminal, start the development web server:
   ```bash
   ./dev.sh start-web
   ```

This approach provides:
- ✅ Live reloading (changes reflect immediately)
- ✅ No cache issues
- ✅ Faster iteration cycle

### For Full Stack Development
```bash
./dev.sh start-all
```

### When You Need to Rebuild Web Service
```bash
./dev.sh rebuild-web
```

## Testing the formatGeneratedResponse Function

### Function Purpose
The `formatGeneratedResponse` function handles the rendering of AI-generated responses, specifically:
- Detects and formats Mermaid diagrams
- Formats markdown-style text (bold, italic, code)
- Handles mixed content (text + diagrams)

### Key Features
1. **Mermaid Diagram Detection**: Uses regex to find `\`\`\`mermaid...` blocks
2. **Text Formatting**: Converts markdown to HTML
3. **Mixed Content**: Handles text before/after diagrams
4. **Fallback**: Plain text formatting if no diagrams found

### Testing
1. Open test file: `http://localhost:3000/test-format-function.html`
2. Check browser console for test results
3. Verify both Mermaid diagram and plain text formatting

## Common Issues & Solutions

### Issue: Changes not reflecting in browser
**Solutions:**
1. Hard refresh (Cmd+Shift+R / Ctrl+Shift+R)
2. Clear browser cache
3. Check if using correct URL (Docker: :3000, Dev server: :3000)
4. Verify volume mounting is working: `docker compose config`

### Issue: Docker build cache
**Solution:**
```bash
docker compose build --no-cache web
```

### Issue: Port conflicts
**Solutions:**
1. Check running services: `lsof -i :3000`
2. Stop conflicting services
3. Use different port in docker-compose.yml

### Issue: Function not found errors
**Likely causes:**
1. JavaScript syntax errors (check browser console)
2. Missing function definitions
3. Scope issues (ensure functions are in correct class/object)

## Browser Developer Tools Tips

1. **Network Tab**: Check if files are being cached (304 vs 200 status)
2. **Console Tab**: Look for JavaScript errors
3. **Application Tab**: Clear storage/cache if needed
4. **Sources Tab**: Verify correct file content is loaded

## Best Practices for Frontend Development

1. **Always test in incognito/private mode** to avoid cache issues
2. **Use browser dev tools** to debug issues
3. **Test with hard refresh** before assuming code issues
4. **Use the development server** for active frontend work
5. **Version control changes** before testing different approaches
6. **Test cross-browser compatibility** for production features

## File Structure

```
web/
├── index.html              # Main web interface
├── Dockerfile              # Production container setup
├── dev-server.py           # Development server
└── test-format-function.html  # Function testing page
```

## Production vs Development

### Production (Docker)
- Uses Nginx to serve static files
- Optimized for performance
- Requires rebuilds for changes
- Use: `./dev.sh start-all`

### Development (Python server)
- Simple HTTP server
- Live reloading
- No caching
- Use: `./dev.sh start-web`

Choose the appropriate mode based on your current development needs.
