# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: services.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import message_types_pb2 as message__types__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='services.proto',
  package='br.edu.ifpb.lib',
  syntax='proto3',
  serialized_options=_b('\n\017br.edu.ifpb.libP\001'),
  serialized_pb=_b('\n\x0eservices.proto\x12\x0f\x62r.edu.ifpb.lib\x1a\x1bgoogle/protobuf/empty.proto\x1a\x13message-types.proto2\x85\x01\n\x0f\x44ocumentService\x12r\n\x19\x66indFrequencyDistribution\x12-.br.edu.ifpb.lib.FrequencyDistributionRequest\x1a&.br.edu.ifpb.lib.FrequencyDistribution2\xb4\x01\n\x03Lda\x12]\n\x1dtreinarModeloLDAComDocumentos\x12\".br.edu.ifpb.lib.DocumentoConteudo\x1a\x16.google.protobuf.Empty(\x01\x12N\n\x10treinarModeloLDA\x12\".br.edu.ifpb.lib.DocumentoConteudo\x1a\x16.google.protobuf.Empty2\xd1\x01\n\x0cRecomendacao\x12\x62\n\x13\x62uscarRecomendacoes\x12$.br.edu.ifpb.lib.RecomendacaoRequest\x1a%.br.edu.ifpb.lib.RecomendacaoResponse\x12]\n\x1dremoverDocumentoDoTreinamento\x12$.br.edu.ifpb.lib.RecomendacaoRequest\x1a\x16.google.protobuf.EmptyB\x13\n\x0f\x62r.edu.ifpb.libP\x01\x62\x06proto3')
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,message__types__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR._options = None

_DOCUMENTSERVICE = _descriptor.ServiceDescriptor(
  name='DocumentService',
  full_name='br.edu.ifpb.lib.DocumentService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=86,
  serialized_end=219,
  methods=[
  _descriptor.MethodDescriptor(
    name='findFrequencyDistribution',
    full_name='br.edu.ifpb.lib.DocumentService.findFrequencyDistribution',
    index=0,
    containing_service=None,
    input_type=message__types__pb2._FREQUENCYDISTRIBUTIONREQUEST,
    output_type=message__types__pb2._FREQUENCYDISTRIBUTION,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_DOCUMENTSERVICE)

DESCRIPTOR.services_by_name['DocumentService'] = _DOCUMENTSERVICE


_LDA = _descriptor.ServiceDescriptor(
  name='Lda',
  full_name='br.edu.ifpb.lib.Lda',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  serialized_start=222,
  serialized_end=402,
  methods=[
  _descriptor.MethodDescriptor(
    name='treinarModeloLDAComDocumentos',
    full_name='br.edu.ifpb.lib.Lda.treinarModeloLDAComDocumentos',
    index=0,
    containing_service=None,
    input_type=message__types__pb2._DOCUMENTOCONTEUDO,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='treinarModeloLDA',
    full_name='br.edu.ifpb.lib.Lda.treinarModeloLDA',
    index=1,
    containing_service=None,
    input_type=message__types__pb2._DOCUMENTOCONTEUDO,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_LDA)

DESCRIPTOR.services_by_name['Lda'] = _LDA


_RECOMENDACAO = _descriptor.ServiceDescriptor(
  name='Recomendacao',
  full_name='br.edu.ifpb.lib.Recomendacao',
  file=DESCRIPTOR,
  index=2,
  serialized_options=None,
  serialized_start=405,
  serialized_end=614,
  methods=[
  _descriptor.MethodDescriptor(
    name='buscarRecomendacoes',
    full_name='br.edu.ifpb.lib.Recomendacao.buscarRecomendacoes',
    index=0,
    containing_service=None,
    input_type=message__types__pb2._RECOMENDACAOREQUEST,
    output_type=message__types__pb2._RECOMENDACAORESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='removerDocumentoDoTreinamento',
    full_name='br.edu.ifpb.lib.Recomendacao.removerDocumentoDoTreinamento',
    index=1,
    containing_service=None,
    input_type=message__types__pb2._RECOMENDACAOREQUEST,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_RECOMENDACAO)

DESCRIPTOR.services_by_name['Recomendacao'] = _RECOMENDACAO

# @@protoc_insertion_point(module_scope)
