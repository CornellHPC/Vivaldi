import urllib.request, bz2, lzma, os, sys
import numpy as np

# Functions related to downloading, extracting, and preparing the datasets
def progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r[%s] %d%%" % ('#' * (percent // 2), percent))
    sys.stdout.flush()

def print_file_size(url):
  req = urllib.request.Request(url, method='HEAD')
  with urllib.request.urlopen(req) as response:
    file_size = response.getheader('Content-Length')
    if file_size:
      file_size = int(file_size)
      print(f"File size: {file_size} bytes")
    else:
      print("Could not retrieve file size")

def download_extract(url) -> str:
  tar_file = "data/" + os.path.basename(url)
  base_file, extension = os.path.splitext(os.path.basename(tar_file))
  txt_out = "data/" + base_file + ".txt"
  if extension not in [".bz2", ".xz"]:
    raise ValueError(f"Unknown file format: {extension}")
  print(f"Downloading {url}...")
  print_file_size(url)
  urllib.request.urlretrieve(url, tar_file, reporthook=progress_hook)
  print()
  print(f"Downloaded {url} to {tar_file}...")
  print(f"Extracting {url}...")
  if extension == ".bz2":
    with open(tar_file, "rb") as f:
      data = bz2.decompress(f.read())
    with open(txt_out, "wb+") as f:
      f.write(data)
  elif extension == ".xz":
    with open(tar_file, "rb") as f:
      data = lzma.decompress(f.read())
    with open(txt_out, "wb+") as f:
      f.write(data)
  os.remove(tar_file)
  print(f"Extracted {url} to {txt_out}...")
  return txt_out

def read_data(formatted_txt_file) -> np.ndarray:
  path = formatted_txt_file
  data = []
  labels = []

  # this currently assumes no missing data, 
  # all formatted with exactly one space between features
  with open(path, 'r') as file:
    rows = [line.rstrip() for line in file]
    for features in rows:
      features = features.split(' ')

      # extract label, which is unused
      label = features.pop(0)
      labels.append(label)

      features = np.float32(
          list(map(lambda feature: float(feature.split(':')[1]), features)))
      data.append(features)

  return np.array(data)
  
MAX_NUM_POINTS = 1000000 ## one million points limit
  
def dump_raw_data(formatted_txt_file):
  print(f"Reading text data from {formatted_txt_file}...")
  base_file, extension = os.path.splitext(formatted_txt_file)
  assert extension == ".txt"
  
  nbytes = os.path.getsize(formatted_txt_file)
  print(f"Byte count is {nbytes}...")
  
  # this currently assumes no missing data, 
  # all formatted with exactly one space between features
  out_bin_file = base_file + ".bin"
  print(f"Writing binary data to {out_bin_file}...")
  lines_read = 0
  with open(out_bin_file, "wb+") as out:
    with open(formatted_txt_file, 'r') as file:
      while True:
        line = file.readline()
        lines_read += 1
        if not line:
          break
        progress_hook(file.tell(), 1, nbytes)
        features = line.rstrip().split(' ')
        features.pop(0)  # extract label, which is unused
        features_as_str = list(map(lambda feature: float(feature.split(':')[1]), features))
        features = np.float32(features_as_str)
        out.write(features.tobytes())
        if lines_read >= MAX_NUM_POINTS:
          print(f"Reached maximum number of points: {MAX_NUM_POINTS}...")
          break
  print()
  print(f"Wrote binary data to {out_bin_file}...")
  
# Step 1: Download and extract the datasets
os.makedirs("data", exist_ok=True)
URLS = [
  "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/poker.t.bz2",
  "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.scale.xz",
  "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.xz"
]
formatted_txt_files = [download_extract(url) for url in URLS]
# Step 2: Clean the datasets
# formatted_txt_files = ["data/HIGGS.txt", "data/australian.txt", "data/poker.t.txt"]
# formatted_txt_files = ["data/mnist8m.scale.txt"]
for file in formatted_txt_files:
  dump_raw_data(file)
