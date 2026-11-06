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
    sh('ssh frame "echo frame1 | sudo -S bash -c \'nohup python3 -I -u /home/steamos/frame-dsp/qurt_boot_test.py --fw /home/steamos/frame-dsp/p_h.mbn --seconds 3 --pause 0.1 --hold > /home/steamos/frame-dsp/hold.out 2>&1 < /dev/null &\' >/dev/null 2>&1"',30)
    time.sleep(20)
    b1=bootid()
    if not b1: b1=wait_up()
    if b1!=b0:
        st='RESET'
    else:
        o=sh('ssh frame "tail -4 frame-dsp/steps.log"',30)
        st='REACHED' if '01eeffc0' in o else 'STALLED'
        sh('ssh frame "echo frame1 | sudo -S systemctl reboot"',30); time.sleep(10)
    print(idx,S[idx],st,flush=True); return st
if __name__=='__main__':
    for i in sys.argv[1:]: test(int(i))
