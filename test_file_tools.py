"""测试文件工具功能"""
import asyncio
import os
from agents.file_tools import HymemFileParser


async def test_file_parser():
    """测试文件解析器"""
    parser = HymemFileParser()
    
    print("=" * 60)
    print("Testing Hymem File Parser")
    print("=" * 60)
    
    test_cases = [
        ("test.pdf", "PDF文件"),
        ("test.csv", "CSV文件"),
        ("test.xlsx", "Excel文件"),
        ("test.docx", "Word文档"),
        ("test.txt", "文本文件"),
        ("test.html", "HTML文件"),
    ]
    
    for file_path, desc in test_cases:
        print(f"\n{'='*60}")
        print(f"测试: {desc} ({file_path})")
        print('='*60)
        
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            continue
        
        try:
            result = await parser.parse_file(file_path)
            
            if result['success']:
                print(f"✅ 解析成功")
                print(f"📊 元数据: {result['metadata']}")
                content_preview = result['content'][:300] if len(result['content']) > 300 else result['content']
                print(f"📄 内容预览:\n{content_preview}")
                if len(result['content']) > 300:
                    print(f"... (总长度: {len(result['content'])} 字符)")
            else:
                print(f"❌ 解析失败: {result['content']}")
                
        except Exception as e:
            print(f"❌ 异常: {e}")
    
    print(f"\n{'='*60}")
    print("测试完成")
    print('='*60)


async def test_with_tool_manager():
    """测试通过BMToolManager使用文件工具"""
    from agents.bm_tools import BMToolManager
    
    print("\n" + "=" * 60)
    print("测试 BMToolManager 文件工具集成")
    print("=" * 60)
    
    manager = BMToolManager()
    
    test_file = "test.pdf"
    if os.path.exists(test_file):
        print(f"\n解析文件: {test_file}")
        result = await manager.parse_file(test_file)
        print(f"成功: {result.get('success', False)}")
        print(f"内容长度: {len(result.get('content', ''))}")
    else:
        print(f"⚠️  测试文件不存在: {test_file}")
    
    await manager.close()


if __name__ == "__main__":
    asyncio.run(test_file_parser())
    asyncio.run(test_with_tool_manager())

