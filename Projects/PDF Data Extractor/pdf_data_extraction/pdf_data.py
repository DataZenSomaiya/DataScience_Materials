from pdf2image import convert_from_path
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'D:/Tesseract/tesseract.exe'

'''
extracting image through pdf2image for multiple pdfs at same time

# def pdf_img(pdf_path,num):
#     # Store Pdf with convert_from_path function
#     images = convert_from_path(pdf_path, 350,poppler_path='C:/Program Files/poppler-0.68.0/bin')
#     for j in range(1,num):
#         for i in range(len(images)):
#             # Save pages as images in the pdf
#             images[i].save('pdf_'+str(j)+'_page' + str(i) + '.jpg', 'JPEG')
'''

def pdf_img(pdf_path):
    # Store Pdf with convert_from_path function
    images = convert_from_path(pdf_path, 350,poppler_path='C:/Program Files/poppler-0.68.0/bin')
    for i in range(len(images)):
        # Save pages as images in the pdf
        images[i].save('page' + str(i) + '.jpg', 'JPEG')


def ocr(image_path):
    # convert the image to black and white for better OCR
    image=cv2.imread(image_path)
    ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    # pytesseract image to string to get results
    text = str(pytesseract.image_to_string(thresh1, config='--psm 6'))
    return text



'''
For conversion of multiple pdfs to images
# solution=[]
# folder='D:/SNEHA/TY/Data Science/Week 1/Invoice Images/'
# count=1
# num=int(input('Enter number of pdfs you want to convert:'))
# for filename in os.listdir(folder):
#     data_from_img=ocr(os.path.join(folder,filename))
#     solution.append(data_from_img)
#     count+=1

'''


def create_file(img_pth):
    data=ocr(img_pth)
    file=open('Ocr_data.txt','w')
    file.write(str(data))
    file.close()
