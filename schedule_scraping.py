import asyncio
from crawler import BajajFinservKnowledgeExtractor

async def main():
    extractor = BajajFinservKnowledgeExtractor()
    try:
        scraped_content, knowledge_graph = await extractor.run_complete_pipeline()
        print(f"Scraped {len(scraped_content)} pages.")
        print(f"Knowledge graph updated with {knowledge_graph.get_statistics()['num_entities']} entities "
              f"and {knowledge_graph.get_statistics()['num_relationships']} relationships.")
    finally:
        await extractor.close()

if __name__ == "__main__":
    asyncio.run(main())