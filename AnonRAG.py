import os
import re

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core import (
    Settings, SimpleDirectoryReader, VectorStoreIndex,
    StorageContext, PromptTemplate
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter

import numpy as np
import pandas as pd
from sphericalKmeans.von_mises_fisher_mixture import VonMisesFisherMixture

from llama_index.core.workflow import Event
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)


class RetrieverEvent(Event):
    """Ids of retrieved nodes and their cluster ids"""

    node_ids: list[str]
    clst_labels: list[int]
    dict_clstid_docid_filename: dict[int, dict[str, str]]


# In[3]:


class AnonymizerEvent(Event):
    """Ids of retrieved nodes and their cluster ids"""
    other_nodes: dict[int, list[str]]
    nodes_in_cluster: dict[int, list[str]]


# In[4]:


class Anontxt(BaseModel):
    """Data model for generating anonymized text."""

    Anonymized_text: str = Field(
        description="Anomnynmized text"
    )


# In[10]:


class AnonymizedRAGWorkflow(Workflow):
    def __init__(self, folder_name: str, file_name: str,
                 openai_api_key: str, nvidia_api_key: str,
                 retrieval_query: str,
                 top_k: int = 10,
                 chunk_size: int = 126,
                 chunk_overlap: int = 20,
                 n_clusters: int = 30,
                 concentration_thresh: int = 3000,
                 verbose=True,
                 **kwargs):
        super().__init__(**kwargs)

        # Set up OpenAI llm and Nvidia embedding
        self.openai_api_key = openai_api_key
        self.nvidia_api_key = nvidia_api_key
        self.llm = OpenAI(model='gpt-4o',
                          api_key=self.openai_api_key,
                          temperature=0.0,
                          timeout=600)
        self.embed_model = NVIDIAEmbedding(api_key=self.nvidia_api_key)
        # Set up other args
        # query that we want to pass anonymized source documents
        self.retrieval_query = retrieval_query
        # folder name that contains source data for retrieval
        self.folder_name = folder_name
        # file name to carry out the anonymized RAG
        self.file_name = file_name
        # how many text chunks to retrieve
        self.top_k = top_k
        # chunks size for splitting texts
        self.chunk_size = chunk_size
        # overlap between chunks
        self.chunk_overlap = chunk_overlap
        # how many clusters to fit the source file text data
        self.n_clusters = n_clusters
        # threshold for the concentration parameter
        self.concentration_thresh = concentration_thresh
        # whether to print out messages or not
        self.verbose = verbose

    @step
    async def retrievenodes(
            self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        "From the supplied query and metadata filters, start the retrieval"
        query = self.retrieval_query
        file_name = self.file_name
        folder_name = self.folder_name
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        embed_model = self.embed_model
        verbose = self.verbose
        top_k = self.top_k

        if self.verbose:
            print(f"""Carry out the retrieval using the following query:\n\n{query}\n\n
            for the file:{file_name} in folder {folder_name}""")

        documents = SimpleDirectoryReader(folder_name).load_data(num_workers=4)
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        index = VectorStoreIndex.from_documents(documents,
                                                transformations=[splitter],
                                                embed_model=embed_model)

        # carry out retrieval of relevant chunks for the query for the supplied file
        metadatafilters = MetadataFilters(
            filters=[ExactMatchFilter(key='file_name', value=file_name)]
        )
        retriever = index.as_retriever(similarity_top_k=top_k, filters=metadatafilters)
        nodes = retriever.retrieve(query)

        # obtain ids of retrieved text chunks
        node_ids = [nd.id_ for nd in nodes]
        if self.verbose:
            print(f"""Top {top_k} relevant chunks for the query retrieved! 
                  Proceed to clustering of documents under the folder {folder_name}""")

        # Obtain a list of lists with content
        # [[index_id,file_name,embedding_vector]]
        # for each index_id
        embdgs = [[key, index.docstore.docs.get(key).metadata['file_name']] + \
                  index.vector_store.get(key) for key in index.docstore.docs.keys()]
        embdgs = pd.DataFrame(embdgs)
        if self.verbose:
            print(f"Clustering documents, this may take a while...")

        # Carry out Spherical Clustering using von Mises-Fisher Mixture Model
        skm = VonMisesFisherMixture(n_clusters=self.n_clusters, posterior_type='soft')
        X = np.array(embdgs.iloc[:, 2:])
        skm.fit(X)

        # Obtain cluster label for each retrieved text chunk to be anonymized for the subsequent AnonRAG task
        clst_labels = [int(skm.labels_[embdgs.iloc[:, 0] == node_id][0]) for node_id in node_ids]

        # Obtain a dictonary of dictionary with cluster labels as the first layer keys
        # and document id as the second layer keys with the file name as the value
        # that is something like {'0':'documentid':'EarningsCalldoc1.pdf',....}
        dict_clstid_docid_filename = {}
        for c_idx in np.unique(skm.labels_):
            c_idx = int(c_idx)
            dict_clstid_docid_filename[c_idx] = {}
            for idx, key in enumerate(embdgs.iloc[:, 0]):
                dict_clstid_docid_filename[c_idx][key] = embdgs.iloc[idx, 1]

        await ctx.set("index", index)
        await ctx.set('skm', skm)
        return (RetrieverEvent(node_ids=node_ids,
                               clst_labels=clst_labels,
                               dict_clstid_docid_filename=dict_clstid_docid_filename))

    @step
    async def groupnodes(
            self, ctx: Context, ev: RetrieverEvent
    ) -> AnonymizerEvent:
        node_ids = ev.node_ids
        clst_labels = ev.clst_labels
        dict_clstid_docid_filename = ev.dict_clstid_docid_filename
        index = await ctx.get('index')
        skm = await ctx.get('skm')

        # dictionary of cluster membership for text chunks from the file
        nodes_in_cluster = {}
        # dictionary of cluster membership for text chunks from other documents in the folder
        # which will be used for the anonymization task on the chunks from the file
        other_nodes = {}
        for idx, val in enumerate(set(clst_labels)):
            nodes_in_cluster[val] = [node_ids[i] for i in range(len(clst_labels)) if clst_labels[i] == val]
            # This part can be made better
            other_nodes[val] = [x for x in dict_clstid_docid_filename[val]. \
                keys() if x not in nodes_in_cluster \
                                and dict_clstid_docid_filename[val][x] not in \
                                [dict_clstid_docid_filename[val][nd_id] for \
                                 nd_id in nodes_in_cluster[val]]]

        await ctx.set("index", index)
        await ctx.set('skm', skm)

        return (AnonymizerEvent(other_nodes=other_nodes,
                                nodes_in_cluster=nodes_in_cluster))

    @step
    async def anonymizenodes(
            self, ctx: Context, ev: AnonymizerEvent
    ) -> StopEvent | None:
        concentration_thresh = self.concentration_thresh
        nodes_in_cluster = ev.nodes_in_cluster
        other_nodes = ev.other_nodes
        index = await ctx.get('index')
        skm = await ctx.get('skm')

        if self.verbose:
            print(f"""Carry out anonymization of relevant text chunks from file:\n{self.file_name}.""")
        txts = {}
        txts['clustered_text_chunks'] = {}
        txts['index'] = index
        txts['cluster_obj'] = skm
        # Text chunk anonymization is done as follows:
        # 1. If the cluster in which the chunk belongs to have high concentration
        # (= other chunks in the same cluster have similar narrative), we
        # swap sensitive attributes to achieve anonymization (pertrubative anonymization)
        # 2. If the cluster in which the chunk belongs to have low concerntrarion,
        # we merge sensitive attributes of the chunk with those other in the same cluster
        # or simply suppress the attributes (nonperturbative anonymization)
        # See the prompt below for details
        for ky in nodes_in_cluster.keys():
            idx_clst = VectorStoreIndex([index.docstore.docs.get(x) for x in other_nodes[ky]],
                                        embed_model=self.embed_model)
            retriever_clst = idx_clst.as_retriever(similarity_top_k=self.top_k)
            ref_txts = ["\n".join([val.get_content() for val \
                                   in retriever_clst.retrieve(index.docstore.docs.get(nd).get_content())]) \
                        for nd in nodes_in_cluster[ky]]
            ref_txts_ids = [[val.id_ for val in retriever_clst.retrieve(index.docstore.docs.get(nd).get_content())] \
                            for nd in nodes_in_cluster[ky]]
            if len(other_nodes[ky]) == 0:
                print('No other files in this cluster, proceed with hard anonymization')
                # TODO write a prompt for this part,
                continue
            else:
                if skm.concentrations_[ky] < concentration_thresh:
                    if self.verbose:
                        print(f'Non-perturbative anonymization in progress')
                    # carry out non-perturbative anonymization

                    Nonperturbative_Annonymization_Template = """
                    You are tasked to modify sensitive information from the suppied original text
                    so that it's not possible to guess company name and the year the document is made.

                    In carrying out this task, you MAY use these reference texts below when coming up with
                    the label for recoding (definition given below). Otherwise, you can carry out the suppression 
                    (definition give below) which does not require you to refer to reference texts. 

                    ----
                    Definitions: 

                    Recoding reduces the amount of information in the sensitive attributes in the 
                    original text by using contextual information of the text. For example, sensitive attributes
                    such as soneone's name could be replaced by their position name instead. To get betteer contextual
                    understanding of the original text, you may use reference texts and carry out recoding.

                    Suppression mask the sensitive attributes in the original text. For exaple, we can suppress 
                    web URL such as https:/www.google.com/ as https:/www.****.com/.

                    -----
                    Reference:
                    {reference_texts}

                    -----
                    Original text chunk to be anonymized:
                    {text_chunk}

                    -----
                    Anonymized text chunk: 

                    """
                    qa_template = PromptTemplate(Nonperturbative_Annonymization_Template)
                    anon_txts = [{'Anonymization_method': 'Non-Perturbative',
                                  'Id': index.docstore.docs.get(nd).id_,
                                  'Anonymized_txt': self.llm.structured_predict(Anontxt,
                                                                                qa_template,
                                                                                reference_texts=ref_txts[idx],
                                                                                text_chunk=index.docstore.docs.get(
                                                                                    nd).get_content()),
                                  'ref_txts_Ids': ref_txts_ids[idx]} for idx, nd in enumerate(nodes_in_cluster[ky])]
                    txts['clustered_text_chunks'][ky] = anon_txts
                else:
                    # carry out perturbative anonymization
                    if self.verbose:
                        print(f'Perturbative anonymization in progress')
                    Perturbative_Annonymization_Template = """
                    You are tasked to modify sensitive information from the suppied original text
                    so that it's not possible to the guess company name and the year the document is made.

                    In carrying out this task, you MUST use these refrerence texts bdlow and modify sensitive attributes
                    in original text data. Specifically, you MUST extract similar sensitive attributes in reference
                    texts and swaps their positions with sensitive attributes in the original text. Below 
                    is an example of this operation.

                    Example: 
                    Original text to be anonymized: 
                    Our exclusive handset agreements with Apple on iPhone has contributed to a 
                    revenue growth in our company by 20 percent compared to the previous quater.

                    Reference texts:
                    Compared to our competitors, the exclusivity deal on Palm Pre, has not attracted
                    enough customers to our carrier business.

                    Our exclusive partnership with Research In Motion. Limited on their product line, 
                    in particular, BlackBerry Storm has contributed to an increase in operating margin by 10%.

                    Anonymized text:
                    Our exclusive handset agreements with  Research In Motion. Limited on Palm Pre has contributed to a 
                    revenue growth in our company by 20 percent compared to the previous quater.

                    -----
                    Reference:
                    {reference_texts}

                    -----
                    Original text chunk to be anonymized:
                    {text_chunk}

                    -----
                    Anonymized text chunk: 

                    """
                    qa_template = PromptTemplate(Perturbative_Annonymization_Template)
                    {}
                    anon_txts = [{'Anonymization_method': 'Perturbative',
                                  'Id': index.docstore.docs.get(nd).id_,
                                  'Anonymized_txt': self.llm.structured_predict(Anontxt,
                                                                                qa_template,
                                                                                reference_texts=ref_txts[idx],
                                                                                text_chunk=index.docstore.docs.get(
                                                                                    nd).get_content()),
                                  'ref_txts_Ids': ref_txts_ids[idx]} for idx, nd in enumerate(nodes_in_cluster[ky])]
                    txts['clustered_text_chunks'][ky] = anon_txts
        return (StopEvent(result=txts))

