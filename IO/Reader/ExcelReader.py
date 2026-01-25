import pandas as pd
from .BaseReader import BaseReader

class ExcelReader(BaseReader):

    def read(self, file_path: str):
        sheets = pd.read_excel(file_path, sheet_name=None)

        structure = []
        combined_text = []

        for name, df in sheets.items():
            text = df.astype(str).values.flatten().tolist()
            combined_text.extend(text)

            structure.append({
                "sheet": name,
                "rows": df.shape[0],
                "cols": df.shape[1],
                "preview": df.head(5).to_dict()
            })

        return {
            "source": "file",
            "file_type": "excel",
            "file_path": file_path,
            "content": "\n".join(combined_text),
            "structure": structure,
            "metadata": {
                "num_sheets": len(sheets),
                "length": len(combined_text)
            }
        }
