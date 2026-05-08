import sys
import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text

if __name__ == '__main__':
    pdf_file = 'AL2002_LabProject.pdf'
    try:
        text = read_pdf(pdf_file)
        with open('pdf_text.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Success")
    except Exception as e:
        print(f"Error: {e}")
