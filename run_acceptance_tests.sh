#!/bin/bash
# Run all acceptance tests from Tasks.md

set -e

echo "🧪 Running AI Video Pipeline Acceptance Tests"
echo "============================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"
command -v ffmpeg >/dev/null 2>&1 || { echo -e "${RED}ffmpeg is required${NC}"; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo -e "${RED}ffprobe is required${NC}"; exit 1; }
command -v pytest >/dev/null 2>&1 || { echo -e "${RED}pytest is required${NC}"; exit 1; }

# Phase 0 Tests
echo -e "\n${YELLOW}Phase 0: Foundation Tests${NC}"
echo "=========================="

echo -e "\n📋 Task 2: Concurrent Database Operations"
pytest tests/test_concurrent_db.py::TestConcurrentDatabase::test_concurrent_writes_no_deadlock -xvs -n 4
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: Concurrent DB writes without deadlock${NC}"
else
    echo -e "${RED}❌ FAIL: Concurrent DB test failed${NC}"
    exit 1
fi

echo -e "\n📋 Task 3: Checkpoint Recovery"
pytest tests/test_checkpoint_recovery.py::TestCheckpointRecovery::test_resume_after_analysis_crash -xvs
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: Resume from checkpoint after crash${NC}"
else
    echo -e "${RED}❌ FAIL: Checkpoint recovery test failed${NC}"
    exit 1
fi

# Phase 1 Tests
echo -e "\n${YELLOW}Phase 1: Core Processing Tests${NC}"
echo "==============================="

echo -e "\n📋 Task 6: Processing Performance"
pytest tests/test_performance_requirements.py::TestPerformanceRequirements::test_processing_time_requirement -xvs
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: Processing < 1.2x source duration${NC}"
else
    echo -e "${RED}❌ FAIL: Processing too slow${NC}"
    exit 1
fi

echo -e "\n📋 Task 7: Filter String Length"
pytest tests/test_performance_requirements.py::TestPerformanceRequirements::test_filter_string_length -xvs
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: Filter string < 300 chars for 50 segments${NC}"
else
    echo -e "${RED}❌ FAIL: Filter string too long${NC}"
    exit 1
fi

echo -e "\n📋 Task 8: Audio Loudness Normalization"
pytest tests/test_performance_requirements.py::TestPerformanceRequirements::test_audio_loudness_spread -xvs
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: Audio spread ≤ 1.5 LU${NC}"
else
    echo -e "${RED}❌ FAIL: Audio spread too high${NC}"
    exit 1
fi

echo -e "\n📋 Task 9: Color Space Compliance"
pytest tests/test_performance_requirements.py::TestPerformanceRequirements::test_color_space_output -xvs
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: Output is BT.709${NC}"
else
    echo -e "${RED}❌ FAIL: Wrong color space${NC}"
    exit 1
fi

# Phase 2 Tests
echo -e "\n${YELLOW}Phase 2: Intelligence & Hardening Tests${NC}"
echo "======================================="

echo -e "\n📋 Task 11: Budget Control"
pytest tests/test_performance_requirements.py::TestPerformanceRequirements::test_budget_limit -xvs
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASS: Budget limit enforced < $5${NC}"
else
    echo -e "${RED}❌ FAIL: Budget control failed${NC}"
    exit 1
fi

# Final Summary
echo -e "\n${YELLOW}Summary${NC}"
echo "======="
echo -e "${GREEN}All acceptance tests passed! ✅${NC}"
echo ""
echo "Next steps:"
echo "1. Run the full litmus test: python litmus_test.py"
echo "2. Deploy monitoring: cd monitoring && ./deploy.sh"
echo "3. Process a real video to validate end-to-end flow"