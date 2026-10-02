"""One synchronized raw-frame slot shared with the NVENC subprocess.

The producer overwrites this slot only after the encoder returns the previous
access unit. QP data and frame IDs retain the existing framed pipe protocol.
"""
import mmap
import os


class SharedFrame:
    def __init__(self, size):
        self.size=size
        self.fd=os.memfd_create('satc-raw-frame',os.MFD_CLOEXEC)
        try:
            os.ftruncate(self.fd,size)
            self.memory=mmap.mmap(self.fd,size,flags=mmap.MAP_SHARED,
                                  prot=mmap.PROT_READ|mmap.PROT_WRITE)
        except BaseException:
            os.close(self.fd)
            raise

    def copy_from(self, raw):
        if len(raw)!=self.size:
            raise ValueError('Unexpected raw frame size for shared encoder slot')
        self.memory[:]=raw

    def close(self):
        self.memory.close()
        os.close(self.fd)
