#!/bin/bash

# OpenGuidance Production Deployment Script
# Author: Nik Jois <nikjois@llamasearch.ai>

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.prod"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Functions
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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        log_warning "Environment file $ENV_FILE not found. Creating template..."
        create_env_template
    fi
    
    log_success "Prerequisites check completed"
}

create_env_template() {
    cat > "$ENV_FILE" << EOF
# OpenGuidance Production Environment Configuration
# Author: Nik Jois <nikjois@llamasearch.ai>

# Database
POSTGRES_PASSWORD=your_secure_postgres_password_here

# Monitoring
GRAFANA_PASSWORD=your_secure_grafana_password_here

# OpenAI API (Optional)
OPENAI_API_KEY=your_openai_api_key_here

# Security
JWT_SECRET=your_jwt_secret_here
API_SECRET_KEY=your_api_secret_key_here

# SSL Configuration
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Backup Configuration
BACKUP_S3_BUCKET=openguidance-backups
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
EOF

    log_warning "Please edit $ENV_FILE with your actual configuration values"
    log_warning "Deployment will continue in 10 seconds or press Ctrl+C to abort"
    sleep 10
}

setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    mkdir -p nginx/ssl
    
    if [[ ! -f "nginx/ssl/cert.pem" ]] || [[ ! -f "nginx/ssl/key.pem" ]]; then
        log_warning "SSL certificates not found. Generating self-signed certificates..."
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=openguidance.local"
        
        log_warning "Self-signed certificates generated. Replace with proper certificates for production."
    fi
    
    log_success "SSL setup completed"
}

backup_data() {
    if [[ "$1" == "true" ]]; then
        log_info "Creating backup before deployment..."
        
        mkdir -p "$BACKUP_DIR"
        
        # Backup database
        if docker ps | grep -q openguidance-postgres; then
            docker exec openguidance-postgres pg_dump -U openguidance openguidance > "$BACKUP_DIR/database.sql"
            log_success "Database backup created"
        fi
        
        # Backup volumes
        docker run --rm -v openguidance-data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/data.tar.gz -C /data .
        
        log_success "Data backup completed: $BACKUP_DIR"
    fi
}

deploy_application() {
    log_info "Deploying OpenGuidance application..."
    
    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
    
    # Build application image
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build --no-cache openguidance-api
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    
    log_success "Application deployment completed"
}

wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for database
    log_info "Waiting for PostgreSQL..."
    while ! docker exec openguidance-postgres pg_isready -U openguidance -d openguidance &> /dev/null; do
        sleep 2
    done
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    while ! docker exec openguidance-redis redis-cli ping &> /dev/null; do
        sleep 2
    done
    
    # Wait for API
    log_info "Waiting for OpenGuidance API..."
    while ! curl -f http://localhost:8000/health &> /dev/null; do
        sleep 5
    done
    
    log_success "All services are ready"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # API health check
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Database health check
    if docker exec openguidance-postgres pg_isready -U openguidance -d openguidance &> /dev/null; then
        log_success "Database health check passed"
    else
        log_error "Database health check failed"
        return 1
    fi
    
    # Redis health check
    if docker exec openguidance-redis redis-cli ping &> /dev/null; then
        log_success "Redis health check passed"
    else
        log_error "Redis health check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Create monitoring directories
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    
    # Create Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    # Create basic dashboard
    cat > monitoring/grafana/dashboards/openguidance.json << EOF
{
  "dashboard": {
    "title": "OpenGuidance System Metrics",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
EOF
    
    log_success "Monitoring setup completed"
}

show_deployment_info() {
    log_success "OpenGuidance deployment completed successfully!"
    echo
    echo "Service URLs:"
    echo "  • OpenGuidance API: http://localhost:8000"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Grafana Dashboard: http://localhost:3000"
    echo "  • Prometheus Metrics: http://localhost:9090"
    echo
    echo "Useful commands:"
    echo "  • View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  • Scale services: docker-compose -f $COMPOSE_FILE up -d --scale openguidance-worker=3"
    echo "  • Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "  • Update services: ./scripts/deploy.sh --update"
    echo
    echo "Monitoring:"
    echo "  • Health checks: curl http://localhost:8000/health"
    echo "  • System metrics: curl http://localhost:8000/monitoring/metrics"
    echo "  • Service status: docker-compose -f $COMPOSE_FILE ps"
    echo
}

cleanup_old_images() {
    log_info "Cleaning up old Docker images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old OpenGuidance images (keep last 3)
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}" | \
        grep openguidance | \
        tail -n +4 | \
        awk '{print $3}' | \
        xargs -r docker rmi
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    local backup_enabled=false
    local update_mode=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup)
                backup_enabled=true
                shift
                ;;
            --update)
                update_mode=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --backup    Create backup before deployment"
                echo "  --update    Update existing deployment"
                echo "  --help      Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    echo "=================================================="
    echo "OpenGuidance Production Deployment"
    echo "Author: Nik Jois <nikjois@llamasearch.ai>"
    echo "Environment: $ENVIRONMENT"
    echo "=================================================="
    
    # Pre-deployment steps
    check_prerequisites
    setup_ssl
    setup_monitoring
    
    # Backup if requested
    backup_data "$backup_enabled"
    
    # Deploy application
    deploy_application
    
    # Post-deployment steps
    wait_for_services
    run_health_checks
    
    # Cleanup
    cleanup_old_images
    
    # Show deployment information
    show_deployment_info
    
    log_success "OpenGuidance is now running in production mode!"
}

# Execute main function
main "$@" 