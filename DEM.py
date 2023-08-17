import urllib.request

with open(f'H:\\data\\DEM\\ch.swisstopo.swissalti3d-tmONMqph.csv') as f:
    for line in f.readlines():
        link = line.rstrip('\n')
        print(link)
        print()
        file = link.split('/')[-1]
        print(file)
        #urllib.request.urlretrieve(link, f'H:\\data\\DEM\\{file}')