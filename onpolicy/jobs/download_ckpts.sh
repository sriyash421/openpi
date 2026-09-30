#!/usr/bin/env bash
# Released checkpoints into the cache openpi's maybe_download reads ($OPENPI_DATA_HOME/openpi-assets/checkpoints/<name>).
set -euo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt
source $S/env.sh
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
for c in pi05_base pi05_libero; do
  $S/dl_venv/bin/python -c "
import gcsfs, os
fs = gcsfs.GCSFileSystem(token='anon')
dst = os.path.join(os.environ['OPENPI_DATA_HOME'], 'openpi-assets/checkpoints/$c')
fs.get('openpi-assets/checkpoints/$c/', dst, recursive=True)
print('[download] $c ->', dst)
" &
done
fail=0; for j in $(jobs -p); do wait $j || fail=1; done
du -sh $OPENPI_DATA_HOME/openpi-assets/checkpoints/*
exit $fail
