import os
from diff_match_patch import diff_match_patch
from AnonRAG import AnonymizedRAGWorkflow
query = """
Assign two different sentiment scores described below about the next quarter. 
Scores must be ranging between 0 and 1, and the score 0.5 represent
neutral sentiment and the closer it gets to 0 the more negative it gets,
while the close it gets to 1, the more positive it gets.

Sentiment scores:

1. Mactoeconomic sentiments: how the company view overall health of the economy
in the next quarter. Words to look for are for example: Interest rates, Exchange rates,
GDP, Unemployment rate, Monetary Policy, etc.

2. Microeconomic sentiments: how the compant view the health of the firm-specific in the
next quarter. Words to look for are for example: Operating margning, Return of equity,
Debt-to-equity ratio, Sales growth, Inventory turnover, etc.
'''
"""
async def main():
    w = AnonymizedRAGWorkflow(folder_name = 'EarningsCall',
                          file_name='ADBEQ1 2015.pdf',
                          openai_api_key = os.environ['OPENAI_API_KEY'],
                          nvidia_api_key= os.environ['NVIDIA_API_KEY'],
                          retrieval_query=query,
                          top_k = 10,timeout = None)
    result = await w.run()
    for ky in result['clustered_text_chunks']:
        anontxts = [anontxt['Anonymized_txt'].Anonymized_text for anontxt in result['clustered_text_chunks'][ky]]
        origtxts = [result['index'].docstore.docs.get(chunk['Id']).get_content() for chunk in result['clustered_text_chunks'][ky]]
        for idx in range(len(anontxts)):
            dmp = diff_match_patch()
            patches = dmp.patch_make(origtxts[idx],anontxts[idx])
            diff_txt = dmp.patch_toText(patches)
            print('Original Text:')
            print(origtxts[idx])
            print('Anonymized Text:')
            print(anontxts[idx])
            print('Difference')
            print(diff_txt)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())