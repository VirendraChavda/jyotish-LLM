# CSV document loader

from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(r"C:\Users\viren\Desktop\Anima\Code\Synth\employee_summary.csv")

csv_data = loader.load()
print(csv_data[0])