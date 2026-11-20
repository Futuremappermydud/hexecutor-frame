# SPDX-License-Identifier: LGPL-3.0-or-later
import subprocess,sys,time
from mkhook import build
S=open('tb_first.txt').read().split()
def sh(c,t=120):
    try: return subprocess.run(c,shell=True,capture_output=True,text=True,timeout=t).stdout
    except Exception: return ''
def bootid(): return sh('ssh -o ConnectTimeout=5 -o BatchMode=yes frame "cat /proc/sys/kernel/random/boot_id"',15).strip()
def wait_up():
    for _ in range(80):
        b=bootid()
        if b: return b
        time.sleep(5)
    sys.exit('no frame')
def test(idx):
    b0=wait_up(); time.sleep(3)
    build(int(S[idx],16),'p_h.mbn'); sh('scp -q p_h.mbn frame:frame-dsp/p_h.mbn')
    sh('ssh frame "echo frame1 | sudo -S bash -c \'nohup python3 -I -u /home/steamos/frame-dsp/qurt_boot_test.py --fw /home/steamos/frame-dsp/p_h.mbn --seconds 3 --pause 0.1 > /home/steamos/frame-dsp/hold.out 2>&1 < /dev/null &\' >/dev/null 2>&1"',30)
    time.sleep(15)
    b1=wait_up()
    o=sh('ssh frame "tail -6 frame-dsp/steps.log"',30)
    reset = (b1!=b0)
    closed='closed /dev/cdsp' in o
    hit='01eeffc0' in o
    st='RESET' if reset else ('REACHED' if hit else 'STALLED/no-marker')
    print(idx,S[idx],st,'closed' if closed else '',flush=True)
    return st
if __name__=='__main__':
    lo,hi=int(sys.argv[1]),int(sys.argv[2])
    while hi-lo>1:
        mid=(lo+hi)//2
        st=test(mid)
        if st=='RESET': hi=mid
        else: lo=mid
    print('last no-reset',lo,S[lo],'first reset',hi,S[hi])
