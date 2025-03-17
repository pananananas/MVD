import openxlab
import os

DOWNLOAD_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/omniobject"

openxlab.login(ak=os.getenv('OPENXLAB_ACCESS_KEY'), sk=os.getenv('OPENXLAB_SECRET_KEY'))
print(f'# {"*" * 100}')
print("Logged in")

print(f'# {"*" * 100}')
print("Dataset information:")
openxlab.dataset.info(dataset_repo='omniobject3d/OmniObject3D-New') #Dataset information viewing

print(f'# {"*" * 100}')
print("Dataset download:")
openxlab.dataset.get(dataset_repo='omniobject3d/OmniObject3D-New', target_path=DOWNLOAD_PATH) # Dataset download

print(f'# {"*" * 100}')
print("Dataset file download:")
openxlab.dataset.download(dataset_repo='omniobject3d/OmniObject3D-New',source_path='/README.md', target_path=DOWNLOAD_PATH) #Dataset file download