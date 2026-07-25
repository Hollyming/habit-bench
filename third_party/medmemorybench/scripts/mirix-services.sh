#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/mirix-services.yml"
ENV_FILE="$PROJECT_ROOT/.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

get_compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    elif docker compose version &> /dev/null; then
        echo "docker compose"
    else
        log_error "Neither docker-compose nor docker compose is available."
        exit 1
    fi
}

start_services() {
    check_docker
    local compose_cmd=$(get_compose_cmd)

    log_info "Starting MIRIX PostgreSQL service..."

    if [ -f "$ENV_FILE" ]; then
        $compose_cmd -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d mirix_db
    else
        $compose_cmd -f "$COMPOSE_FILE" up -d mirix_db
    fi

    log_info "Waiting for PostgreSQL to be ready..."
    wait_for_postgres

    log_success "MIRIX PostgreSQL service is running!"
    log_info "Connection: postgresql://mirix:mirix@localhost:5432/mirix"
}

start_all_services() {
    check_docker
    local compose_cmd=$(get_compose_cmd)

    log_info "Starting MIRIX services (PostgreSQL + Redis)..."

    if [ -f "$ENV_FILE" ]; then
        $compose_cmd -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile with-redis up -d
    else
        $compose_cmd -f "$COMPOSE_FILE" --profile with-redis up -d
    fi

    log_info "Waiting for services to be ready..."
    wait_for_postgres
    wait_for_redis

    log_success "All MIRIX services are running!"
    log_info "PostgreSQL: postgresql://mirix:mirix@localhost:5432/mirix"
    log_info "Redis: redis://localhost:6379"
}

stop_services() {
    check_docker
    local compose_cmd=$(get_compose_cmd)

    log_info "Stopping MIRIX services..."
    $compose_cmd -f "$COMPOSE_FILE" --profile with-redis down
    log_success "MIRIX services stopped."
}

check_status() {
    check_docker
    local compose_cmd=$(get_compose_cmd)

    echo ""
    echo "============================================"
    echo "       MIRIX Services Status"
    echo "============================================"
    echo ""

    if docker ps --format '{{.Names}}' | grep -q '^mirix_pgvector$'; then
        log_success "PostgreSQL (mirix_pgvector): Running"

        if docker exec mirix_pgvector pg_isready -U mirix -d mirix > /dev/null 2>&1; then
            log_success "  - Database is accepting connections"
        else
            log_warning "  - Database is starting up..."
        fi
    else
        log_warning "PostgreSQL (mirix_pgvector): Not running"
    fi

    if docker ps --format '{{.Names}}' | grep -q '^mirix_redis$'; then
        log_success "Redis (mirix_redis): Running"

        if docker exec mirix_redis redis-cli ping > /dev/null 2>&1; then
            log_success "  - Redis is accepting connections"
        else
            log_warning "  - Redis is starting up..."
        fi
    else
        log_info "Redis (mirix_redis): Not running (optional)"
    fi

    echo ""
    echo "============================================"
}

view_logs() {
    check_docker
    local compose_cmd=$(get_compose_cmd)

    log_info "Showing MIRIX service logs (Ctrl+C to exit)..."
    $compose_cmd -f "$COMPOSE_FILE" --profile with-redis logs -f
}

reset_data() {
    check_docker
    local compose_cmd=$(get_compose_cmd)

    log_warning "This will DELETE all MIRIX data! Are you sure? (y/N)"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Stopping services and removing volumes..."
        $compose_cmd -f "$COMPOSE_FILE" --profile with-redis down -v
        log_success "All MIRIX data has been reset."
    else
        log_info "Reset cancelled."
    fi
}

wait_for_postgres() {
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker exec mirix_pgvector pg_isready -U mirix -d mirix > /dev/null 2>&1; then
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo ""
    log_error "PostgreSQL failed to start within ${max_attempts} seconds"
    return 1
}

wait_for_redis() {
    local max_attempts=15
    local attempt=1

    if ! docker ps --format '{{.Names}}' | grep -q '^mirix_redis$'; then
        return 0  # Redis not started, skip waiting
    fi

    while [ $attempt -le $max_attempts ]; do
        if docker exec mirix_redis redis-cli ping > /dev/null 2>&1; then
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo ""
    log_warning "Redis failed to start within ${max_attempts} seconds"
    return 1
}

test_connection() {
    check_docker

    log_info "Testing PostgreSQL connection..."

    if ! docker ps --format '{{.Names}}' | grep -q '^mirix_pgvector$'; then
        log_error "PostgreSQL container is not running. Start it first with: $0 start"
        exit 1
    fi

    if docker exec mirix_pgvector psql -U mirix -d mirix -c "SELECT 1;" > /dev/null 2>&1; then
        log_success "PostgreSQL connection successful!"
    else
        log_error "PostgreSQL connection failed!"
        exit 1
    fi

    log_info "Testing pgvector extension..."
    local pgvector_test=$(docker exec mirix_pgvector psql -U mirix -d mirix -t -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';" 2>/dev/null | tr -d ' ')

    if [ -n "$pgvector_test" ]; then
        log_success "pgvector extension installed (version: $pgvector_test)"
    else
        log_error "pgvector extension not found!"
        exit 1
    fi

    log_info "Testing vector operations..."
    docker exec mirix_pgvector psql -U mirix -d mirix -c "
        CREATE TEMP TABLE test_vectors (id serial PRIMARY KEY, embedding vector(3));
        INSERT INTO test_vectors (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
        SELECT * FROM test_vectors ORDER BY embedding <-> '[1,2,3]' LIMIT 1;
        DROP TABLE test_vectors;
    " > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        log_success "Vector operations working correctly!"
    else
        log_error "Vector operations test failed!"
        exit 1
    fi

    if docker ps --format '{{.Names}}' | grep -q '^mirix_redis$'; then
        log_info "Testing Redis connection..."
        if docker exec mirix_redis redis-cli ping > /dev/null 2>&1; then
            log_success "Redis connection successful!"
        else
            log_warning "Redis connection failed"
        fi
    fi

    echo ""
    log_success "All tests passed! MIRIX services are ready for evaluation."
}

show_usage() {
    echo ""
    echo "============================================"
    echo "  MIRIX Services Management Script"
    echo "============================================"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start       Start PostgreSQL service only"
    echo "  start-all   Start PostgreSQL and Redis services"
    echo "  stop        Stop all services"
    echo "  status      Check service status"
    echo "  logs        View service logs"
    echo "  reset       Reset all data (destructive!)"
    echo "  test        Test database connection"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start       # Quick start with PostgreSQL only"
    echo "  $0 start-all   # Start with Redis for better performance"
    echo "  $0 status      # Check if services are running"
    echo "  $0 test        # Verify everything is working"
    echo ""
}

case "${1:-help}" in
    start)
        start_services
        ;;
    start-all)
        start_all_services
        ;;
    stop)
        stop_services
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    reset)
        reset_data
        ;;
    test)
        test_connection
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
