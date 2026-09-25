#!/bin/bash
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=${CHIA_ROOT:-"$PWD"/../../chia}
export CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=2400000
export DSE_AGENT=1
export DSE_AGENT_BUDGET=0          # unlimited: the agent decides
export DSE_OUT="$PWD"/out_loopB_k64
${PYTHON:-python3} -u chia_dse.py run 64 > loop_agent.log 2>&1
echo "EXIT=$?" >> loop_agent.log
