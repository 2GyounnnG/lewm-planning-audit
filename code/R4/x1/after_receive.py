"""Local increment reception -> byte verification -> CPU replay -> per-file seal."""
import argparse,subprocess,sys,tarfile,time
from . import core

def run(folder):
    folder=core.Path(folder).resolve();receipt_path=folder/'RECOVERY_ARCHIVE.json'
    while not receipt_path.exists():time.sleep(10)
    receipt=core.read(receipt_path);archive=folder/core.Path(receipt['archive']['path']).name
    # rsync --partial may install an incomplete file under the final basename
    # after a transport failure. Existence alone is never a completion signal.
    while not archive.exists() or archive.stat().st_size!=receipt['archive']['bytes']:time.sleep(10)
    core.verify(dict(receipt['archive'],path=str(archive)));task=archive.name.removesuffix('_main.tar.gz');bundle=folder/task
    if not bundle.exists():
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar.getmembers():
                p=core.Path(member.name)
                if p.is_absolute() or '..' in p.parts or p.parts[0]!=task or member.issym() or member.islnk():raise RuntimeError('Unsafe recovery member')
            tar.extractall(folder,filter='data')
    core.verify(dict(receipt['manifest'],path=str(bundle/'RECOVERY_MANIFEST.json')))
    subprocess.run([sys.executable,'-u','-m','x1.recovery_cpu',str(bundle)],check=True)
    subprocess.run([sys.executable,'-u','-m','x1.recovery_seal',str(bundle)],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();run(a.folder)
