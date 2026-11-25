#!/usr/bin/env python3
"""
HyMem Evaluation Script

评测HyMem框架在BrowseComp等数据集上的表现。
支持单问题测试和批量评测。
"""

import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.hymem_agent import run_hymem_simple, HyMemAgent
from agents.hymem_memory import HyMemStore, HyMemManager
from agents.bm_tools import BMToolManager


async def eval_single_question(
    question: str,
    expected_answer: str = None,
    search_base_url: str = "http://localhost:8021",
    llm_model: str = "deepseek-chat",
    aux_model: str = "deepseek-chat",
    verbose: bool = True
) -> dict:
    """评测单个问题"""
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question[:200]}...")
        print(f"{'='*60}")
    
    try:
        answer, confidence, stats = await run_hymem_simple(
            problem=question,
            search_base_url=search_base_url,
            llm_model=llm_model,
            aux_model=aux_model
        )
        
        elapsed = time.time() - start_time
        
        result = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "expected_answer": expected_answer,
            "stats": stats,
            "elapsed_time": elapsed,
            "success": bool(answer and len(answer.strip()) > 0)
        }
        
        if expected_answer:
            result["exact_match"] = answer.lower().strip() == expected_answer.lower().strip()
        
        if verbose:
            print(f"\nAnswer: {answer}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Stats: {stats}")
            print(f"Time: {elapsed:.1f}s")
            if expected_answer:
                print(f"Expected: {expected_answer}")
                print(f"Match: {result.get('exact_match', 'N/A')}")
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        if verbose:
            print(f"Error: {e}")
        return {
            "question": question,
            "answer": "",
            "confidence": 0.0,
            "expected_answer": expected_answer,
            "error": str(e),
            "elapsed_time": elapsed,
            "success": False
        }


async def eval_dataset(
    data_path: str,
    output_dir: str = "./scripts/results",
    search_base_url: str = "http://localhost:8021",
    llm_model: str = "deepseek-chat",
    aux_model: str = "deepseek-chat",
    limit: int = None,
    verbose: bool = True
) -> dict:
    """评测数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'data' in data:
        items = data['data']
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    
    if limit:
        items = items[:limit]
    
    print(f"\nEvaluating {len(items)} items from {data_path}")
    
    results = []
    success_count = 0
    
    for i, item in enumerate(items):
        question = item.get('query') or item.get('question') or item.get('problem_statement', '')
        expected = item.get('answer') or item.get('expected_answer', '')
        
        print(f"\n[{i+1}/{len(items)}] Processing...")
        
        result = await eval_single_question(
            question=question,
            expected_answer=expected,
            search_base_url=search_base_url,
            llm_model=llm_model,
            aux_model=aux_model,
            verbose=verbose
        )
        
        result['item_id'] = item.get('id', i)
        results.append(result)
        
        if result.get('success'):
            success_count += 1
    
    summary = {
        "total": len(items),
        "success": success_count,
        "success_rate": success_count / len(items) if items else 0,
        "avg_confidence": sum(r.get('confidence', 0) for r in results) / len(results) if results else 0,
        "avg_time": sum(r.get('elapsed_time', 0) for r in results) / len(results) if results else 0,
        "total_folds": sum(r.get('stats', {}).get('fold_count', 0) for r in results),
        "total_replans": sum(r.get('stats', {}).get('replan_count', 0) for r in results)
    }
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_path = os.path.join(output_dir, f"hymem_results_{timestamp}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    
    summary_path = os.path.join(output_dir, f"hymem_summary_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("HyMem Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Model: {llm_model}\n")
        f.write(f"Aux Model: {aux_model}\n")
        f.write(f"Total: {summary['total']}\n")
        f.write(f"Success: {summary['success']} ({summary['success_rate']:.1%})\n")
        f.write(f"Avg Confidence: {summary['avg_confidence']:.2f}\n")
        f.write(f"Avg Time: {summary['avg_time']:.1f}s\n")
        f.write(f"Total Folds: {summary['total_folds']}\n")
        f.write(f"Total Replans: {summary['total_replans']}\n")
    
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total: {summary['total']}")
    print(f"Success: {summary['success']} ({summary['success_rate']:.1%})")
    print(f"Avg Confidence: {summary['avg_confidence']:.2f}")
    print(f"Avg Time: {summary['avg_time']:.1f}s")
    print(f"Results saved to: {results_path}")
    
    return summary


async def test_memory_generation():
    """测试Memory生成功能"""
    print("\nTesting Memory Generation...")
    
    manager = HyMemManager(aux_model="deepseek-chat")
    
    question = "What is the capital of France?"
    planner_history = """Turn 1: I need to find the capital of France.
Turn 2: Let me search for this information.
Turn 3: I found that Paris is the capital of France."""
    
    tool_calls = [
        {"tool_name": "web_search", "arguments": {"query": "capital of France"}, "response": "Paris is the capital...", "success": True}
    ]
    
    try:
        episode, working, tool = await manager.generate_all_memories(
            question=question,
            planner_history=planner_history,
            recent_history="Turn 3: I found that Paris is the capital of France.",
            tool_call_history=tool_calls
        )
        
        print(f"Episode Memory: {json.dumps(episode, indent=2) if episode else 'None'}")
        print(f"Working Memory: {json.dumps(working, indent=2) if working else 'None'}")
        print(f"Tool Memory: {json.dumps(tool, indent=2) if tool else 'None'}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        await manager.close()


async def test_trigger():
    """测试触发器功能"""
    from agents.hymem_trigger import FoldTrigger, TriggerConfig
    
    print("\nTesting Fold Trigger...")
    
    config = TriggerConfig(
        planner_context_threshold=0.80,
        max_folds=3
    )
    trigger = FoldTrigger(config)
    memory = HyMemStore()
    
    should_fold, reason = trigger.should_fold_planner(
        context_length=25000,
        max_length=32000,
        response="This is a test response",
        memory_store=memory,
        current_turn=5
    )
    print(f"Context 25k/32k: should_fold={should_fold}, reason={reason}")
    
    should_fold, reason = trigger.should_fold_planner(
        context_length=10000,
        max_length=32000,
        response="Let me fold <fold_planner>",
        memory_store=memory,
        current_turn=5
    )
    print(f"Active fold marker: should_fold={should_fold}, reason={reason}")
    
    trigger.record_confidence(0.2)
    trigger.record_confidence(0.2)
    should_fold, reason = trigger.should_fold_planner(
        context_length=10000,
        max_length=32000,
        response="Test",
        memory_store=memory,
        current_turn=5
    )
    print(f"Low confidence x2: should_fold={should_fold}, reason={reason}")
    
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="HyMem Evaluation Script")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "dataset", "test"],
                        help="Evaluation mode")
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to evaluate")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="./scripts/results",
                        help="Output directory for results")
    parser.add_argument("--search_url", type=str, default="http://localhost:8021",
                        help="Search server URL")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat",
                        help="LLM model name")
    parser.add_argument("--aux_model", type=str, default="deepseek-chat",
                        help="Auxiliary model for memory generation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of items to evaluate")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")
    return parser.parse_args()


async def main():
    args = parse_args()
    
    if args.mode == "test":
        print("Running HyMem Tests...")
        await test_trigger()
        await test_memory_generation()
        print("\nTests completed.")
        
    elif args.mode == "single":
        question = args.question or "What is the capital of France?"
        await eval_single_question(
            question=question,
            search_base_url=args.search_url,
            llm_model=args.llm_model,
            aux_model=args.aux_model,
            verbose=args.verbose
        )
        
    elif args.mode == "dataset":
        if not args.data_path:
            print("Error: --data_path required for dataset mode")
            return
        await eval_dataset(
            data_path=args.data_path,
            output_dir=args.output_dir,
            search_base_url=args.search_url,
            llm_model=args.llm_model,
            aux_model=args.aux_model,
            limit=args.limit,
            verbose=args.verbose
        )


if __name__ == "__main__":
    asyncio.run(main())
