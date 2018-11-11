import os
import zipfile
try:
    from urllib.request  import urlopen
except ImportError:
    from urllib2 import urlopen


def download_data(url):

	u = urlopen(url)
	data = u.read()
	u.close()
 
	os.chdir('..')
	path = 'data/raw/'

	with open(path+ 'chess_data.zip', "wb") as f :
		f.write(data)

	zip_ref = zipfile.ZipFile(path+'chess_data.zip', 'r')
	zip_ref.extractall(path)
	zip_ref.close()


	os.remove(path+'chess_data.zip')
	os.mkdir(path + 'Chess ID Public Data/output_test/empty')

if __name__ == '__main__':
	url = "https://www.dropbox.com/s/8gj7yzu4n3oqm81/Chess%20ID%20Public%20Data.zip?dl=1" 
	print('Downloading ...')
	download_data(url)
	print('Done.')
