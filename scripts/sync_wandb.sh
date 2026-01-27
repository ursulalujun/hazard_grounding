#!/bin/bash
# WandB Sync Script - Run on a node with internet access
# This script syncs offline wandb logs to wandb.ai every 30 minutes

# ==============================================================================
# Configuration
# ==============================================================================

# Directory containing wandb offline logs (shared storage path)
WANDB_DIR="/mnt/shared-storage-user/luxiaoya/code/EAI/SafePlanner/risk_grounding/wandb"

# Sync interval in seconds (30 minutes = 1800 seconds)
SYNC_INTERVAL=${SYNC_INTERVAL:-1800}

# wandb project name (optional, auto-detected if not specified)
WANDB_PROJECT="${WANDB_PROJECT:-}"

# wandb entity (optional, your username or team name)
WANDB_ENTITY="${WANDB_ENTITY:-}"

# Log file for sync operations
LOG_FILE="${WANDB_DIR}/sync_log.txt"

# ==============================================================================
# Functions
# ==============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

sync_wandb() {
    log "Starting wandb sync..."

    # Check if wandb directory exists
    if [ ! -d "$WANDB_DIR" ]; then
        log "Warning: wandb directory not found: $WANDB_DIR"
        return 1
    fi

    # Find all wandb offline run directories
    RUN_DIRS=$(find "$WANDB_DIR" -type d -name "offline-*" 2>/dev/null)

    if [ -z "$RUN_DIRS" ]; then
        log "No offline runs found in $WANDB_DIR"
        return 0
    fi

    # Sync each offline run
    SYNCED_COUNT=0
    FAILED_COUNT=0

    while IFS= read -r run_dir; do
        log "Syncing: $run_dir"

        # Build sync command
        SYNC_CMD="wandb sync \"$run_dir\""

        if [ -n "$WANDB_PROJECT" ]; then
            SYNC_CMD="$SYNC_CMD --project \"$WANDB_PROJECT\""
        fi

        if [ -n "$WANDB_ENTITY" ]; then
            SYNC_CMD="$SYNC_CMD --entity \"$WANDB_ENTITY\""
        fi

        # Execute sync
        if eval $SYNC_CMD; then
            log "Successfully synced: $run_dir"
            SYNCED_COUNT=$((SYNCED_COUNT + 1))
        else
            log "Failed to sync: $run_dir"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done <<< "$RUN_DIRS"

    log "Sync complete: $SYNCED_COUNT succeeded, $FAILED_COUNT failed"
    return 0
}

# ==============================================================================
# Main Loop
# ==============================================================================

log "=========================================="
log "WandB Sync Service Started"
log "WandB Directory: $WANDB_DIR"
log "Sync Interval: $SYNC_INTERVAL seconds ($((SYNC_INTERVAL / 60)) minutes)"
log "=========================================="

# Run sync once on startup
sync_wandb

# Main sync loop
while true; do
    sleep "$SYNC_INTERVAL"
    sync_wandb
done
