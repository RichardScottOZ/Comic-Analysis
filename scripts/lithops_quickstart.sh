#!/bin/bash
# CoSMo PSS Lithops Quick Start Script

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         CoSMo PSS Lithops Integration Quick Start            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

print_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
print_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
print_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }
print_warning() { echo -e "\033[0;33m[WARNING]\033[0m $1"; }

print_info "Checking Lithops installation..."
if ! python -c "import lithops" 2>/dev/null; then
    print_warning "Lithops not found. Install with: pip install lithops[aws]"
    exit 1
fi
print_success "Lithops is installed"

print_info "Use this script to quickly set up Lithops for CoSMo PSS"
print_info "See documentation/optimization/Lithops_Integration.md for full details"
echo ""
print_success "Ready to use Lithops! 🚀"
