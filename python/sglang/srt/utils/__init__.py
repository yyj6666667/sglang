# Temporarily do this to avoid changing all imports in the repo
from sglang.srt.utils.common import *
# Re-export network helpers so legacy "from sglang.srt.utils import get_local_ip_auto" works
# after #20646 split utils/network.py out of utils/common.py.
from sglang.srt.utils.network import *
