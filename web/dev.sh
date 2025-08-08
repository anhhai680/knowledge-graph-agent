#!/bin/bash

# Development script for Knowledge Graph Agent
# This script helps with common development tasks

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$PROJECT_ROOT/web"

show_help() {
    echo "Knowledge Graph Agent Development Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start-all     Start all services with Docker (includes rebuild)"
    echo "  start-web     Start only the web interface with live reloading"
    echo "  start-backend Start only backend services (API, Chroma, Memgraph)"
    echo "  rebuild-web   Force rebuild and restart the web service"
    echo "  stop          Stop all Docker services"
    echo "  logs          Show logs from all services"
    echo "  help          Show this help message"
    echo ""
    echo "For development, use 'start-backend' + 'start-web' for faster iteration."
}

start_all() {
    echo "🔄 Starting all services..."
    cd "$PROJECT_ROOT"
    docker compose down
    docker compose build web
    docker compose up -d
    echo "✅ All services started!"
    echo "📱 Web interface: http://localhost:3000"
    echo "🔧 API: http://localhost:8000"
}

start_web() {
    echo "🌐 Starting web development server..."
    echo "📱 Web interface will be available at: http://localhost:3000"
    echo "💡 This server supports live reloading - changes will be reflected immediately"
    echo ""
    cd "$WEB_DIR"
    python3 dev-server.py
}

start_backend() {
    echo "⚡ Starting backend services only..."
    cd "$PROJECT_ROOT"
    docker compose up -d app chroma memgraph
    echo "✅ Backend services started!"
    echo "🔧 API: http://localhost:8000"
    echo "💾 Chroma DB: http://localhost:8001"
    echo "🗃️ Memgraph: bolt://localhost:7687"
}

rebuild_web() {
    echo "🔨 Rebuilding web service..."
    cd "$PROJECT_ROOT"
    docker compose down web
    docker compose build web
    docker compose up -d web
    echo "✅ Web service rebuilt and restarted!"
}

stop_all() {
    echo "🛑 Stopping all services..."
    cd "$PROJECT_ROOT"
    docker compose down
    echo "✅ All services stopped!"
}

show_logs() {
    echo "📋 Showing logs from all services..."
    cd "$PROJECT_ROOT"
    docker compose logs -f
}

case "$1" in
    "start-all")
        start_all
        ;;
    "start-web")
        start_web
        ;;
    "start-backend")
        start_backend
        ;;
    "rebuild-web")
        rebuild_web
        ;;
    "stop")
        stop_all
        ;;
    "logs")
        show_logs
        ;;
    "help"|"")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
