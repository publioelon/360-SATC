#!/usr/bin/env python3
"""Replay SATC's H.264 stream through the retained GXRStream FIFO path."""
import argparse,os,subprocess,sys,tempfile,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Annex-B .h264 stream from satc.py encode')
    p.add_argument('--host',required=True)
    p.add_argument('--port',type=int,default=9001)
    p.add_argument('--python',default=sys.executable)
    args=p.parse_args()
    if not args.input.is_file():p.error('Encoded input file does not exist')
    sender_path=Path(__file__).resolve().parent/'ubuntu/sender/webrtc_sender.py'
    with tempfile.TemporaryDirectory(prefix='satc-stream-') as directory:
        fifo=Path(directory)/'encoded.h264';os.mkfifo(fifo)
        env=os.environ.copy();env['QGXS_EXTERNAL_H264_FIFO']=str(fifo)
        sender=subprocess.Popen([args.python,str(sender_path),'h264',str(args.input.resolve()),args.host,str(args.port),'4096','2048','60','12000'],env=env)
        feeder=None
        try:
            feeder=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','warning','-re','-r','60','-i',str(args.input.resolve()),'-an','-c:v','copy','-f','h264','-y',str(fifo)])
            started=time.monotonic()
            # Fail promptly if the receiver connection or sender setup fails.
            while feeder.poll() is None:
                if sender.poll() is not None:
                    raise RuntimeError(f'Sender exited with status {sender.returncode}')
                if time.monotonic()-started>3600:
                    raise RuntimeError('Replay timed out')
                try:feeder.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            if feeder.returncode:raise RuntimeError(f'FFmpeg exited with status {feeder.returncode}')
            return 0
        finally:
            for process in (feeder,sender):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':raise SystemExit(main())
