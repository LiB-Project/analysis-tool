import pickle

import gensim
import grpc
from concurrent import futures
import time
import logging

import yaml
from tika import parser
from lda_model_generator import LdaModelGenerator

# import the generated classes:
import python_service_pb2_grpc
import python_service_pb2
import extractor
from util import Util

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_MAX_LENGTH_MESSAGE_GRPC = 10485760


class DocumentServiceImpl(python_service_pb2_grpc.DocumentServiceServicer):
    def findFrequencyDistribution(self, request, context):
        idDocument = request.document.idDocument
        file = request.document.file
        totalPalavras = request.total
        # Reading document
        parsed = parser.from_buffer(file)
        content = parsed['content']
        return extractor.generate_frequency_distribution(id=idDocument, content=content, total=totalPalavras)


class LdaServiceImpl(python_service_pb2_grpc.LdaServicer):

    def treinarModeloLDA(self, request, context):
        generator = LdaModelGenerator()
        generator.execute(id_doc=request.idDocumento, conteudo=request.conteudo)
        return python_service_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def treinarModeloLDAComDocumentos(self, request_iterator, context):
        for documentContent in request_iterator:
            print(f"idDocumento={documentContent.idDocumento}, conteudo{documentContent.conteudo}")
        return python_service_pb2.google_dot_protobuf_dot_empty__pb2.Empty()


class RecomendacaoServiceImpl(python_service_pb2_grpc.RecomendacaoServicer):

    def buscarRecomendacoes(self, request, context):
        # Load configuration
        with open('config.yml') as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
        fp.close()

        docs_preprocessed_dict = Util.read_docs_preprocessed()
        dictionary = Util.load_dictionary()
        docs_bow = Util.read_docs_bow()
        lda_model = Util.load_lda_model()

        doc = docs_preprocessed_dict[request.idDocument]
        doc_bow = docs_bow[request.idDocument]

        lda_index = gensim.similarities.MatrixSimilarity(lda_model[docs_bow.values()])
        similarities = lda_index[lda_model[doc_bow]]
        similarities = sorted(enumerate(similarities), key=lambda item: -item[1])

        recomendacaoList = []
        for similar in similarities:
            idDocument = list(docs_preprocessed_dict.keys())[similar[0]]
            similaridade = similar[1]
            recomendacao = python_service_pb2.RecomendacaoDocumento(idDocument=idDocument, similaridade=similaridade)
            recomendacaoList.append(recomendacao)
        response = python_service_pb2.RecomendacaoResponse(recomendacoes=recomendacaoList)
        return response

    def removerDocumentoDoTreinamento(self, request, context):
        if Util.docs_preprocessed_exists():
            docs_preprocessed = Util.read_docs_preprocessed()
            del docs_preprocessed[request.idDocument]
            Util.save_docs_preprocessed(docs_preprocessed)

        return python_service_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

def serve():
    options = [
        ('grpc.max_send_message_length', _MAX_LENGTH_MESSAGE_GRPC),
        ('grpc.max_receive_message_length', _MAX_LENGTH_MESSAGE_GRPC)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    python_service_pb2_grpc.add_DocumentServiceServicer_to_server(DocumentServiceImpl(), server)
    python_service_pb2_grpc.add_LdaServicer_to_server(LdaServiceImpl(), server)
    python_service_pb2_grpc.add_RecomendacaoServicer_to_server(RecomendacaoServiceImpl(), server)
    server.add_insecure_port('[::]:50051')
    print('Starting server. Listening on port 50051.')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()