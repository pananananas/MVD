# Usage
#     python3 download_collection.py -o <collection_owner> -c <collection_name> [-d <download_directory>]
#     python3 download_collection.py -o "GoogleResearch" -c "Scanned Objects by Google Research"
# Description
#     This script will download all models contained within a collection.
#     Link to the collection:
#     https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research
#
import sys, json, requests, os
import getopt

if sys.version_info[0] < 3:
    raise Exception("Python 3 or greater is required. Try running `python3 download_collection.py`")

collection_name = ''
owner_name = ''
download_dir = '/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/gso/'

# Read options
optlist, args = getopt.getopt(sys.argv[1:], 'o:c:d:')

sensor_config_file = ''
private_token = ''
for o, v in optlist:
    if o == "-o":
        owner_name = v.replace(" ", "%20")
    if o == "-c":
        collection_name = v.replace(" ", "%20")
    if o == "-d":
        download_dir = v

if not owner_name:
    print('Error: missing `-o <owner_name>` option')
    quit()

if not collection_name:
    print('Error: missing `-c <collection_name>` option')
    quit()

# Create the download directory if it doesn't exist
try:
    print(f"Using download directory: {download_dir}")
    os.makedirs(download_dir, exist_ok=True)
    print(f"Directory is ready: {os.path.exists(download_dir)}")
except Exception as e:
    print(f"Error creating download directory: {e}")
    quit()

print("Downloading models from the {}/{} collection.".format(owner_name, collection_name.replace("%20", " ")))

page = 1
count = 0

# The Fuel server URL.
base_url ='https://fuel.gazebosim.org/'

# Fuel server version.
fuel_version = '1.0'

# Path to get the models in the collection
next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)

# Path to download a single model in the collection
download_url = base_url + fuel_version + '/{}/models/'.format(owner_name)

# Iterate over the pages
while True:
    url = base_url + fuel_version + next_url

    # Get the contents of the current page.
    r = requests.get(url)

    if not r or not r.text:
        break

    # Convert to JSON
    models = json.loads(r.text)

    # Compute the next page's URL
    page = page + 1
    next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)
  
    # Download each model 
    for model in models:
        count+=1
        model_name = model['name']
        print(f'Downloading ({count}) {model_name}')
        
        # Create full path for file
        file_path = os.path.join(download_dir, f"{model_name}.zip")
        
        try:
            download = requests.get(download_url+model_name+'.zip', stream=True)
            download.raise_for_status()  # Raise exception for HTTP errors
            
            with open(file_path, 'wb') as fd:
                for chunk in download.iter_content(chunk_size=1024*1024):
                    fd.write(chunk)
            print(f'Successfully saved to {file_path}')
        except requests.exceptions.RequestException as e:
            print(f'Error downloading {model_name}: {e}')
        except IOError as e:
            print(f'Error saving {model_name}: {e}')

print(f'Done. Downloaded {count} models to {download_dir}')
