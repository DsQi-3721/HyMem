"""
GAIA Benchmark 文件工具集成示例
展示如何在评测中使用文件解析工具
"""
import asyncio
import os
from pathlib import Path
from agents.bm_tools import BMToolManager


async def process_gaia_question_with_file(
    question: str,
    file_path: str = None,
    gaia_dataset_dir: str = None
):
    """
    处理包含文件的GAIA问题
    
    Args:
        question: GAIA问题文本
        file_path: 相对文件路径（如 "2023/validation/xxx.pdf"）
        gaia_dataset_dir: GAIA数据集根目录
    """
    tool_manager = BMToolManager()
    
    print(f"问题: {question}")
    
    if file_path:
        abs_path = os.path.join(gaia_dataset_dir or ".", file_path)
        
        if os.path.exists(abs_path):
            print(f"\n解析附件: {file_path}")
            
            result = await tool_manager.parse_file(abs_path)
            
            if result['success']:
                print(f"✅ 文件解析成功")
                print(f"📊 元数据: {result['metadata']}")
                
                file_content = result['content']
                
                if len(file_content) > 2000:
                    file_summary = file_content[:2000] + "\n[文件内容已截断...]"
                else:
                    file_summary = file_content
                
                enhanced_question = f"""{question}

[附件内容摘要]:
{file_summary}

请基于附件内容回答问题。"""
                
                print(f"\n增强后的问题长度: {len(enhanced_question)} 字符")
                return enhanced_question, file_content
            else:
                print(f"❌ 文件解析失败: {result['content']}")
                return question, None
        else:
            print(f"⚠️  文件不存在: {abs_path}")
            return question, None
    else:
        print("ℹ️  无附件（text-only问题）")
        return question, None
    
    await tool_manager.close()


async def example_gaia_workflow():
    """示例：完整的GAIA评测流程"""
    
    gaia_dataset_dir = "/data1/test/xq/dataset/GAIA"
    
    example_items = [
        {
            "question": "根据附件中的数据表格，2023年第三季度的销售额是多少？",
            "file_path": "2023/validation/sales_data.csv"
        },
        {
            "question": "这篇论文的作者是谁？发表于哪一年？",
            "file_path": "2023/validation/research_paper.pdf"
        },
        {
            "question": "What is the capital of France?",
            "file_path": None
        }
    ]
    
    for i, item in enumerate(example_items, 1):
        print(f"\n{'='*60}")
        print(f"示例 {i}")
        print('='*60)
        
        enhanced_q, file_content = await process_gaia_question_with_file(
            question=item['question'],
            file_path=item['file_path'],
            gaia_dataset_dir=gaia_dataset_dir
        )
        
        print(f"\n处理结果:")
        print(f"- 原始问题长度: {len(item['question'])}")
        print(f"- 增强问题长度: {len(enhanced_q)}")
        print(f"- 文件内容可用: {file_content is not None}")


async def integrate_with_hymem_agent():
    """示例：在HyMemAgent中集成文件工具"""
    
    print("\n" + "="*60)
    print("HyMemAgent 文件工具集成示例")
    print("="*60)
    
    gaia_question = "根据附件PDF中的表格，找出销售额最高的产品"
    file_path = "/path/to/gaia/2023/validation/sales.pdf"
    
    if os.path.exists(file_path):
        tool_manager = BMToolManager()
        
        file_result = await tool_manager.parse_file(file_path)
        
        if file_result['success']:
            problem_with_file = f"""{gaia_question}

[附件内容]:
{file_result['content'][:1500]}
"""
            print("✅ 文件内容已注入到问题上下文")
            print(f"上下文长度: {len(problem_with_file)}")
        
        await tool_manager.close()
    else:
        print(f"⚠️  示例文件不存在: {file_path}")


if __name__ == "__main__":
    print("GAIA Benchmark 文件工具集成示例\n")
    
    asyncio.run(example_gaia_workflow())
    asyncio.run(integrate_with_hymem_agent())

