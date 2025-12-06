#!/bin/bash

set -eo pipefail

# Error handler function
handle_error() {
  local exit_code=$1
  local error_message="$2"
  log "Error: $error_message (Exit Code: $exit_code)" "error"
  exit "$exit_code"
}

# Check sudo
if [ "$EUID" -ne 0 ]; then
  handle_error 1 "Cannot escalate privileges. Run the script as root or via sudo."
fi

# Logger Function
log() {
  local message="$1"
  local type="$2"
  local endcolor="\033[0m"
  local color
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  case "$type" in
  "info") color="\033[38;5;79m" ;;
  "success") color="\033[1;32m" ;;
  "error") color="\033[1;31m" ;;
  *) color="\033[1;34m" ;;
  esac

  echo -e "${color}${timestamp} - ${message}${endcolor}"
}

# Function to check for command availability
command_exists() {
  command -v "$1" &>/dev/null
}

check_os() {
  if [ -f /etc/debian_version ]; then
    type=deb
  elif [ -f /etc/redhat-release ]; then
    type=rpm
  else
    echo "Error: Unsupported operating system."
    exit 1
  fi

  arch_type=$(uname -m)
  if [ "$arch_type" == "x86_64" ]; then
    arch="amd64"
  elif [ "$arch_type" == "aarch64" ]; then
    arch="arm64"
  else
    handle_error 1 "Unsupported architecture: $arch_type. Only amd64 and arm64 supported."
  fi
}

check_postgres() {
  if command_exists psql; then
    psql_version=$(psql -V | awk '{print $3}' | cut -d'.' -f1)
    case "$psql_version" in
    14 | 15 | 16 | 17)
      log "PostgreSQL version $psql_version detected."
      ;;
    *)
      handle_error 1 "Unsupported PostgreSQL version: $psql_version. Only versions 14, 15, 16, and 17 are supported."
      ;;
    esac
  else
    handle_error 1 "Couldn't find PostgreSQL installation."
  fi
}

# Function to install the pre-requisites for Debian systems
install_deb_prereqs() {
  log "Installing pre-requisites..." "info"

  # Update the system package index
  if ! apt-get update; then
    handle_error 1 "Failed to run 'apt-get update'."
  fi

  # Install the required packages
  if ! apt-get install -y apt-transport-https gpg wget libecpg-dev; then
    handle_error 1 "Failed to install prerequisite packages."
  fi
}

# Function to install the prerequisites for Red Hat systems
install_rpm_prereqs() {
  log "Installing pre-requisites..." "info"

  # Update the system package index
  if ! dnf makecache; then
    handle_error 1 "Failed to run 'dnf makecache'."
  fi

  # Install the required packages
  if ! dnf install -y dnf-plugins-core gpg wget libecpg; then
    handle_error 1 "Failed to install prerequisite packages."
  fi
}

# Function to configure the DEB repo
configure_install_deb() {
  rm -f /etc/apt/keyrings/packages.activeloop.gpg
  rm -f /etc/apt/sources.list.d/activeloop.list

  if ! wget -qO- https://packages.activeloop.io/keys/activeloop.asc | gpg --dearmor >packages.activeloop.gpg; then
    handle_error 1 "Failed to download the Activeloop signing key."
  fi

  if ! install -D -o root -g root -m 644 packages.activeloop.gpg /etc/apt/keyrings/packages.activeloop.gpg; then
    handle_error 1 "Failed to set the key with correct permissions."
  fi

  rm -f packages.activeloop.gpg

  echo "deb [arch=${arch} signed-by=/etc/apt/keyrings/packages.activeloop.gpg] https://packages.activeloop.io/deb/pg-deeplake stable main" |
    tee /etc/apt/sources.list.d/activeloop.list >/dev/null

  if ! apt-get update; then
    handle_error 1 "Failed to run 'apt-get update'."
  else
    log "Repository configured successfully."
  fi

  if ! apt-get install -y pg-deeplake-"$psql_version"; then
    handle_error 1 "Failed to install the extension."
  else
    log "Deep Lake extension v$psql_version installed successfully."
    log "To activate it, run in psql: CREATE EXTENSION IF NOT EXISTS pg_deeplake;" "info"
  fi
}

# Function to configure the RPM repo
configure_install_rpm() {
  rm -f /etc/yum.repos.d/activeloop.repo

  if ! rpm --import https://packages.activeloop.io/keys/activeloop.asc; then
    handle_error 1 "Failed to import the Activeloop signing key."
  fi

  echo -e "[activeloop]\nname=Activeloop\nbaseurl=https://packages.activeloop.io/rpm\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.activeloop.io/keys/activeloop.asc" |
    tee /etc/yum.repos.d/activeloop.repo >/dev/null

  if ! dnf makecache; then
    handle_error 1 "Failed to run 'dnf makecache'."
  else
    log "Repository configured successfully."
  fi

  if ! dnf install -y pg-deeplake-"$psql_version"; then
    handle_error 1 "Failed to install the extension."
  else
    log "Deep Lake extension v$psql_version installed successfully."
    log "To activate it, run in psql: CREATE EXTENSION IF NOT EXISTS pg_deeplake;" "info"
  fi
}

# Checks
check_os
check_postgres

# Main execution
if [ "$type" == "deb" ]; then
  install_deb_prereqs || handle_error 1 "Failed installing pre-requisites."
  configure_install_deb || handle_error 1 "Failed configuring repository and installing the package."
elif [ "$type" == "rpm" ]; then
  install_rpm_prereqs || handle_error 1 "Failed installing pre-requisites."
  configure_install_rpm || handle_error 1 "Failed configuring repository and installing the package."
fi
