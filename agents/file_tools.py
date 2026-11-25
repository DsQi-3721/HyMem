"""
Hymem File Tools - 本地文件解析工具（GAIA Text-only支持）
精简自DeepResearch file_tools，纯本地化实现
"""
import os
import re
import hashlib
import json
from typing import Dict, Any, Optional, List
from pathlib import Path


def hash_file(file_path: str) -> str:
    """计算文件SHA256哈希"""
    hasher = hashlib.sha256()
    hasher.update(file_path.encode())
    with open(file_path, 'rb') as f:
        hasher.update(f.read(4096))
    return hasher.hexdigest()


def detect_file_type(file_path: str) -> str:
    """检测文件类型"""
    ext = Path(file_path).suffix.lower().lstrip('.')
    type_map = {
        'pdf': 'pdf',
        'csv': 'csv',
        'tsv': 'tsv',
        'xlsx': 'xlsx',
        'xls': 'xls',
        'docx': 'docx',
        'doc': 'docx',
        'txt': 'txt',
        'html': 'html',
        'htm': 'html',
        'json': 'txt',
        'jsonl': 'txt',
        'xml': 'xml',
    }
    return type_map.get(ext, 'txt')


def clean_text(text: str) -> str:
    """清理文本"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def table_to_markdown(table: List[List]) -> str:
    """表格转markdown"""
    if not table or not table[0]:
        return ""
    lines = []
    for row in table:
        cleaned = [str(cell).replace('\n', ' ') if cell is not None else '' for cell in row]
        lines.append('| ' + ' | '.join(cleaned) + ' |')
    return '\n'.join(lines)


class HymemFileParser:
    """轻量级文件解析器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or "/tmp/hymem_file_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, file_path: str) -> str:
        """获取缓存文件路径"""
        cache_key = hash_file(file_path)
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_cache(self, file_path: str) -> Optional[Dict]:
        """加载缓存"""
        cache_path = self._get_cache_path(file_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def _save_cache(self, file_path: str, data: Dict):
        """保存缓存"""
        cache_path = self._get_cache_path(file_path)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except:
            pass
    
    async def parse_file(self, file_path: str) -> Dict[str, Any]:
        """统一文件解析接口（异步版本）"""
        if not os.path.exists(file_path):
            return {"content": f"[Error: File not found: {file_path}]", "metadata": {}}
        
        cached = self._load_cache(file_path)
        if cached:
            return cached
        
        file_type = detect_file_type(file_path)
        
        try:
            if file_type == 'pdf':
                result = self._parse_pdf(file_path)
            elif file_type in ('csv', 'tsv'):
                result = self._parse_csv(file_path, sep=',' if file_type == 'csv' else '\t')
            elif file_type in ('xlsx', 'xls'):
                result = self._parse_excel(file_path)
            elif file_type == 'docx':
                result = self._parse_docx(file_path)
            elif file_type == 'html':
                result = self._parse_html(file_path)
            elif file_type == 'xml':
                result = self._parse_xml(file_path)
            else:
                result = self._parse_text(file_path)
        except Exception as e:
            result = {
                "content": f"[Error parsing {file_type} file: {str(e)}]",
                "metadata": {"error": str(e)}
            }
        
        self._save_cache(file_path, result)
        return result
    
    def _parse_pdf(self, path: str) -> Dict:
        """PDF解析"""
        try:
            import pdfplumber
        except ImportError:
            return {"content": "[Error: pdfplumber not installed. Run: pip install pdfplumber]", "metadata": {}}
        
        text_parts = []
        total_pages = 0
        
        try:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = []
                    
                    text = page.extract_text()
                    if text and text.strip():
                        page_text.append(text.strip())
                    
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_md = table_to_markdown(table)
                            if table_md:
                                page_text.append(f"\n[Table]\n{table_md}\n")
                    
                    if page_text:
                        text_parts.append(f"=== Page {page_num} ===\n" + "\n".join(page_text))
        except Exception as e:
            return {"content": f"[Error: PDF parsing failed: {str(e)}]", "metadata": {}}
        
        content = clean_text("\n\n".join(text_parts))
        return {
            "content": content,
            "metadata": {"pages": total_pages, "file_type": "pdf", "length": len(content)}
        }
    
    def _parse_csv(self, path: str, sep: str = ',') -> Dict:
        """CSV/TSV解析"""
        try:
            import pandas as pd
            from tabulate import tabulate
        except ImportError:
            return {"content": "[Error: pandas or tabulate not installed]", "metadata": {}}
        
        try:
            df = pd.read_csv(path, sep=sep, nrows=1000)
            
            if len(df) > 500:
                content = f"[Dataset Info]\nRows: {len(df)}, Columns: {len(df.columns)}\n"
                content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                content += "[First 100 rows]\n"
                content += tabulate(df.head(100), headers='keys', tablefmt='pipe', showindex=False)
            else:
                content = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            
            return {
                "content": content,
                "metadata": {"rows": len(df), "columns": len(df.columns), "file_type": "csv"}
            }
        except Exception as e:
            return {"content": f"[Error: CSV parsing failed: {str(e)}]", "metadata": {}}
    
    def _parse_excel(self, path: str) -> Dict:
        """Excel解析"""
        try:
            import pandas as pd
            from tabulate import tabulate
        except ImportError:
            return {"content": "[Error: pandas or openpyxl not installed]", "metadata": {}}
        
        try:
            xls = pd.ExcelFile(path)
            parts = [f"[Excel File: {len(xls.sheet_names)} sheets]"]
            
            for sheet_name in xls.sheet_names[:5]:
                df = pd.read_excel(path, sheet_name=sheet_name, nrows=200)
                parts.append(f"\n=== Sheet: {sheet_name} ===")
                parts.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                
                if len(df) > 100:
                    parts.append("[First 100 rows]")
                    parts.append(tabulate(df.head(100), headers='keys', tablefmt='pipe', showindex=False))
                else:
                    parts.append(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
            
            content = "\n".join(parts)
            return {
                "content": content,
                "metadata": {"sheets": len(xls.sheet_names), "file_type": "excel"}
            }
        except Exception as e:
            return {"content": f"[Error: Excel parsing failed: {str(e)}]", "metadata": {}}
    
    def _parse_docx(self, path: str) -> Dict:
        """Word文档解析"""
        try:
            from docx import Document
        except ImportError:
            return {"content": "[Error: python-docx not installed. Run: pip install python-docx]", "metadata": {}}
        
        try:
            doc = Document(path)
            parts = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    parts.append(text)
            
            for table in doc.tables:
                table_rows = []
                for row in table.rows:
                    table_rows.append([cell.text.strip() for cell in row.cells])
                if table_rows:
                    parts.append("\n[Table]\n" + table_to_markdown(table_rows))
            
            content = clean_text("\n\n".join(parts))
            return {
                "content": content,
                "metadata": {"paragraphs": len(doc.paragraphs), "tables": len(doc.tables), "file_type": "docx"}
            }
        except Exception as e:
            return {"content": f"[Error: DOCX parsing failed: {str(e)}]", "metadata": {}}
    
    def _parse_html(self, path: str) -> Dict:
        """HTML解析"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"content": "[Error: beautifulsoup4 not installed]", "metadata": {}}
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
            
            for script in soup(['script', 'style', 'meta', 'link']):
                script.decompose()
            
            text = soup.get_text(separator='\n')
            content = clean_text(text)
            
            return {
                "content": content,
                "metadata": {"file_type": "html", "length": len(content)}
            }
        except Exception as e:
            return {"content": f"[Error: HTML parsing failed: {str(e)}]", "metadata": {}}
    
    def _parse_xml(self, path: str) -> Dict:
        """XML解析"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return {
                "content": content[:50000],
                "metadata": {"file_type": "xml", "length": len(content)}
            }
        except Exception as e:
            return {"content": f"[Error: XML parsing failed: {str(e)}]", "metadata": {}}
    
    def _parse_text(self, path: str) -> Dict:
        """文本文件解析"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return {
                "content": content,
                "metadata": {"file_type": "text", "length": len(content)}
            }
        except Exception as e:
            return {"content": f"[Error: Text file parsing failed: {str(e)}]", "metadata": {}}


async def parse_file_async(file_path: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """异步文件解析接口"""
    parser = HymemFileParser(cache_dir=cache_dir)
    return parser.parse_file(file_path)
