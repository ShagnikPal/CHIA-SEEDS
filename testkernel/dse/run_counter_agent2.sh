#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"
export CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=2400000
${PYTHON:-python3} counter_agent.py run \
  --budget 6 \
  --out "$PWD"/out_counter_agent2 \
  > counter_agent2.log 2>&1
echo "EXIT=$?" >> counter_agent2.log
