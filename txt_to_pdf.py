from fpdf import FPDF
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

with open("sample.txt", "r", encoding="utf-8") as file:
    for line in file:
        pdf.cell(200, 10, txt=line.strip(), ln=True)

pdf.output("sample.pdf")
print("? sample.pdf created.")
