#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"
export CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=2400000
export FORGE_BUDGET=6
${PYTHON:-python3} -u forge_mnk.py > forge_mnk.log 2>&1
echo "EXIT=$?" >> forge_mnk.log
