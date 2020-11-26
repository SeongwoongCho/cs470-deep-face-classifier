from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"방탄소년단 정국, 아이콘 바비, 워너원 박지훈, 엑소 수호, 안형섭, 수지, 트와이스 나연, 트와이스 다현, 이세영, 백진희","limit":1000,"print_urls":True, "format":"jpg"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
