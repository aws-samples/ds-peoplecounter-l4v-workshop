# -*- coding: utf-8 -*-
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. // SPDX-License-Identifier: MIT-0
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference-server.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='inference-server.proto',
  package='AWS.LookoutVision.Edge',
  syntax='proto3',
  serialized_options=b'\n com.amazonaws.lookoutvision.edge',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x16inference-server.proto\x12\x16\x41WS.LookoutVision.Edge\"@\n\x12SharedMemoryHandle\x12\x0c\n\x04size\x18\x01 \x01(\x04\x12\x0e\n\x06offset\x18\x02 \x01(\x04\x12\x0c\n\x04name\x18\x03 \x01(\t\"\x90\x01\n\x06\x42itmap\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\x13\n\tbyte_data\x18\x03 \x01(\x0cH\x00\x12J\n\x14shared_memory_handle\x18\x04 \x01(\x0b\x32*.AWS.LookoutVision.Edge.SharedMemoryHandleH\x00\x42\x06\n\x04\x64\x61ta\"a\n\x16\x44\x65tectAnomaliesRequest\x12\x17\n\x0fmodel_component\x18\x01 \x01(\t\x12.\n\x06\x62itmap\x18\x02 \x01(\x0b\x32\x1e.AWS.LookoutVision.Edge.Bitmap\"?\n\x13\x44\x65tectAnomalyResult\x12\x14\n\x0cis_anomalous\x18\x01 \x01(\x08\x12\x12\n\nconfidence\x18\x02 \x01(\x02\"e\n\x17\x44\x65tectAnomaliesResponse\x12J\n\x15\x64\x65tect_anomaly_result\x18\x01 \x01(\x0b\x32+.AWS.LookoutVision.Edge.DetectAnomalyResult\"\x9a\x01\n\x10ModelDescription\x12\x17\n\x0fmodel_component\x18\x01 \x01(\t\x12 \n\x18lookout_vision_model_arn\x18\x02 \x01(\t\x12\x33\n\x06status\x18\x03 \x01(\x0e\x32#.AWS.LookoutVision.Edge.ModelStatus\x12\x16\n\x0estatus_message\x18\x04 \x01(\t\"\x97\x01\n\rModelMetadata\x12\x17\n\x0fmodel_component\x18\x01 \x01(\t\x12 \n\x18lookout_vision_model_arn\x18\x02 \x01(\t\x12\x33\n\x06status\x18\x03 \x01(\x0e\x32#.AWS.LookoutVision.Edge.ModelStatus\x12\x16\n\x0estatus_message\x18\x04 \x01(\t\",\n\x11StartModelRequest\x12\x17\n\x0fmodel_component\x18\x01 \x01(\t\"I\n\x12StartModelResponse\x12\x33\n\x06status\x18\x01 \x01(\x0e\x32#.AWS.LookoutVision.Edge.ModelStatus\"+\n\x10StopModelRequest\x12\x17\n\x0fmodel_component\x18\x01 \x01(\t\"H\n\x11StopModelResponse\x12\x33\n\x06status\x18\x01 \x01(\x0e\x32#.AWS.LookoutVision.Edge.ModelStatus\"\x13\n\x11ListModelsRequest\"K\n\x12ListModelsResponse\x12\x35\n\x06models\x18\x01 \x03(\x0b\x32%.AWS.LookoutVision.Edge.ModelMetadata\"/\n\x14\x44\x65scribeModelRequest\x12\x17\n\x0fmodel_component\x18\x01 \x01(\t\"\\\n\x15\x44\x65scribeModelResponse\x12\x43\n\x11model_description\x18\x01 \x01(\x0b\x32(.AWS.LookoutVision.Edge.ModelDescription*O\n\x0bModelStatus\x12\x0b\n\x07STOPPED\x10\x00\x12\x0c\n\x08STARTING\x10\x01\x12\x0b\n\x07RUNNING\x10\x02\x12\n\n\x06\x46\x41ILED\x10\x03\x12\x0c\n\x08STOPPING\x10\x04\x32\x9f\x04\n\x0fInferenceServer\x12r\n\x0f\x44\x65tectAnomalies\x12..AWS.LookoutVision.Edge.DetectAnomaliesRequest\x1a/.AWS.LookoutVision.Edge.DetectAnomaliesResponse\x12\x63\n\nStartModel\x12).AWS.LookoutVision.Edge.StartModelRequest\x1a*.AWS.LookoutVision.Edge.StartModelResponse\x12`\n\tStopModel\x12(.AWS.LookoutVision.Edge.StopModelRequest\x1a).AWS.LookoutVision.Edge.StopModelResponse\x12\x63\n\nListModels\x12).AWS.LookoutVision.Edge.ListModelsRequest\x1a*.AWS.LookoutVision.Edge.ListModelsResponse\x12l\n\rDescribeModel\x12,.AWS.LookoutVision.Edge.DescribeModelRequest\x1a-.AWS.LookoutVision.Edge.DescribeModelResponseB\"\n com.amazonaws.lookoutvision.edgeb\x06proto3'
)

_MODELSTATUS = _descriptor.EnumDescriptor(
  name='ModelStatus',
  full_name='AWS.LookoutVision.Edge.ModelStatus',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STOPPED', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='STARTING', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RUNNING', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FAILED', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='STOPPING', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1322,
  serialized_end=1401,
)
_sym_db.RegisterEnumDescriptor(_MODELSTATUS)

ModelStatus = enum_type_wrapper.EnumTypeWrapper(_MODELSTATUS)
STOPPED = 0
STARTING = 1
RUNNING = 2
FAILED = 3
STOPPING = 4



_SHAREDMEMORYHANDLE = _descriptor.Descriptor(
  name='SharedMemoryHandle',
  full_name='AWS.LookoutVision.Edge.SharedMemoryHandle',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='size', full_name='AWS.LookoutVision.Edge.SharedMemoryHandle.size', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='offset', full_name='AWS.LookoutVision.Edge.SharedMemoryHandle.offset', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='AWS.LookoutVision.Edge.SharedMemoryHandle.name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=114,
)


_BITMAP = _descriptor.Descriptor(
  name='Bitmap',
  full_name='AWS.LookoutVision.Edge.Bitmap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='AWS.LookoutVision.Edge.Bitmap.width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='AWS.LookoutVision.Edge.Bitmap.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='byte_data', full_name='AWS.LookoutVision.Edge.Bitmap.byte_data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shared_memory_handle', full_name='AWS.LookoutVision.Edge.Bitmap.shared_memory_handle', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='data', full_name='AWS.LookoutVision.Edge.Bitmap.data',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=117,
  serialized_end=261,
)


_DETECTANOMALIESREQUEST = _descriptor.Descriptor(
  name='DetectAnomaliesRequest',
  full_name='AWS.LookoutVision.Edge.DetectAnomaliesRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_component', full_name='AWS.LookoutVision.Edge.DetectAnomaliesRequest.model_component', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bitmap', full_name='AWS.LookoutVision.Edge.DetectAnomaliesRequest.bitmap', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=263,
  serialized_end=360,
)


_DETECTANOMALYRESULT = _descriptor.Descriptor(
  name='DetectAnomalyResult',
  full_name='AWS.LookoutVision.Edge.DetectAnomalyResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='is_anomalous', full_name='AWS.LookoutVision.Edge.DetectAnomalyResult.is_anomalous', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='AWS.LookoutVision.Edge.DetectAnomalyResult.confidence', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=362,
  serialized_end=425,
)


_DETECTANOMALIESRESPONSE = _descriptor.Descriptor(
  name='DetectAnomaliesResponse',
  full_name='AWS.LookoutVision.Edge.DetectAnomaliesResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='detect_anomaly_result', full_name='AWS.LookoutVision.Edge.DetectAnomaliesResponse.detect_anomaly_result', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=427,
  serialized_end=528,
)


_MODELDESCRIPTION = _descriptor.Descriptor(
  name='ModelDescription',
  full_name='AWS.LookoutVision.Edge.ModelDescription',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_component', full_name='AWS.LookoutVision.Edge.ModelDescription.model_component', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lookout_vision_model_arn', full_name='AWS.LookoutVision.Edge.ModelDescription.lookout_vision_model_arn', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status', full_name='AWS.LookoutVision.Edge.ModelDescription.status', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status_message', full_name='AWS.LookoutVision.Edge.ModelDescription.status_message', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=531,
  serialized_end=685,
)


_MODELMETADATA = _descriptor.Descriptor(
  name='ModelMetadata',
  full_name='AWS.LookoutVision.Edge.ModelMetadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_component', full_name='AWS.LookoutVision.Edge.ModelMetadata.model_component', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lookout_vision_model_arn', full_name='AWS.LookoutVision.Edge.ModelMetadata.lookout_vision_model_arn', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status', full_name='AWS.LookoutVision.Edge.ModelMetadata.status', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status_message', full_name='AWS.LookoutVision.Edge.ModelMetadata.status_message', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=688,
  serialized_end=839,
)


_STARTMODELREQUEST = _descriptor.Descriptor(
  name='StartModelRequest',
  full_name='AWS.LookoutVision.Edge.StartModelRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_component', full_name='AWS.LookoutVision.Edge.StartModelRequest.model_component', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=841,
  serialized_end=885,
)


_STARTMODELRESPONSE = _descriptor.Descriptor(
  name='StartModelResponse',
  full_name='AWS.LookoutVision.Edge.StartModelResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='AWS.LookoutVision.Edge.StartModelResponse.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=887,
  serialized_end=960,
)


_STOPMODELREQUEST = _descriptor.Descriptor(
  name='StopModelRequest',
  full_name='AWS.LookoutVision.Edge.StopModelRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_component', full_name='AWS.LookoutVision.Edge.StopModelRequest.model_component', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=962,
  serialized_end=1005,
)


_STOPMODELRESPONSE = _descriptor.Descriptor(
  name='StopModelResponse',
  full_name='AWS.LookoutVision.Edge.StopModelResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='AWS.LookoutVision.Edge.StopModelResponse.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1007,
  serialized_end=1079,
)


_LISTMODELSREQUEST = _descriptor.Descriptor(
  name='ListModelsRequest',
  full_name='AWS.LookoutVision.Edge.ListModelsRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1081,
  serialized_end=1100,
)


_LISTMODELSRESPONSE = _descriptor.Descriptor(
  name='ListModelsResponse',
  full_name='AWS.LookoutVision.Edge.ListModelsResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='models', full_name='AWS.LookoutVision.Edge.ListModelsResponse.models', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1102,
  serialized_end=1177,
)


_DESCRIBEMODELREQUEST = _descriptor.Descriptor(
  name='DescribeModelRequest',
  full_name='AWS.LookoutVision.Edge.DescribeModelRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_component', full_name='AWS.LookoutVision.Edge.DescribeModelRequest.model_component', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1179,
  serialized_end=1226,
)


_DESCRIBEMODELRESPONSE = _descriptor.Descriptor(
  name='DescribeModelResponse',
  full_name='AWS.LookoutVision.Edge.DescribeModelResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_description', full_name='AWS.LookoutVision.Edge.DescribeModelResponse.model_description', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1228,
  serialized_end=1320,
)

_BITMAP.fields_by_name['shared_memory_handle'].message_type = _SHAREDMEMORYHANDLE
_BITMAP.oneofs_by_name['data'].fields.append(
  _BITMAP.fields_by_name['byte_data'])
_BITMAP.fields_by_name['byte_data'].containing_oneof = _BITMAP.oneofs_by_name['data']
_BITMAP.oneofs_by_name['data'].fields.append(
  _BITMAP.fields_by_name['shared_memory_handle'])
_BITMAP.fields_by_name['shared_memory_handle'].containing_oneof = _BITMAP.oneofs_by_name['data']
_DETECTANOMALIESREQUEST.fields_by_name['bitmap'].message_type = _BITMAP
_DETECTANOMALIESRESPONSE.fields_by_name['detect_anomaly_result'].message_type = _DETECTANOMALYRESULT
_MODELDESCRIPTION.fields_by_name['status'].enum_type = _MODELSTATUS
_MODELMETADATA.fields_by_name['status'].enum_type = _MODELSTATUS
_STARTMODELRESPONSE.fields_by_name['status'].enum_type = _MODELSTATUS
_STOPMODELRESPONSE.fields_by_name['status'].enum_type = _MODELSTATUS
_LISTMODELSRESPONSE.fields_by_name['models'].message_type = _MODELMETADATA
_DESCRIBEMODELRESPONSE.fields_by_name['model_description'].message_type = _MODELDESCRIPTION
DESCRIPTOR.message_types_by_name['SharedMemoryHandle'] = _SHAREDMEMORYHANDLE
DESCRIPTOR.message_types_by_name['Bitmap'] = _BITMAP
DESCRIPTOR.message_types_by_name['DetectAnomaliesRequest'] = _DETECTANOMALIESREQUEST
DESCRIPTOR.message_types_by_name['DetectAnomalyResult'] = _DETECTANOMALYRESULT
DESCRIPTOR.message_types_by_name['DetectAnomaliesResponse'] = _DETECTANOMALIESRESPONSE
DESCRIPTOR.message_types_by_name['ModelDescription'] = _MODELDESCRIPTION
DESCRIPTOR.message_types_by_name['ModelMetadata'] = _MODELMETADATA
DESCRIPTOR.message_types_by_name['StartModelRequest'] = _STARTMODELREQUEST
DESCRIPTOR.message_types_by_name['StartModelResponse'] = _STARTMODELRESPONSE
DESCRIPTOR.message_types_by_name['StopModelRequest'] = _STOPMODELREQUEST
DESCRIPTOR.message_types_by_name['StopModelResponse'] = _STOPMODELRESPONSE
DESCRIPTOR.message_types_by_name['ListModelsRequest'] = _LISTMODELSREQUEST
DESCRIPTOR.message_types_by_name['ListModelsResponse'] = _LISTMODELSRESPONSE
DESCRIPTOR.message_types_by_name['DescribeModelRequest'] = _DESCRIBEMODELREQUEST
DESCRIPTOR.message_types_by_name['DescribeModelResponse'] = _DESCRIBEMODELRESPONSE
DESCRIPTOR.enum_types_by_name['ModelStatus'] = _MODELSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SharedMemoryHandle = _reflection.GeneratedProtocolMessageType('SharedMemoryHandle', (_message.Message,), {
  'DESCRIPTOR' : _SHAREDMEMORYHANDLE,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.SharedMemoryHandle)
  })
_sym_db.RegisterMessage(SharedMemoryHandle)

Bitmap = _reflection.GeneratedProtocolMessageType('Bitmap', (_message.Message,), {
  'DESCRIPTOR' : _BITMAP,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.Bitmap)
  })
_sym_db.RegisterMessage(Bitmap)

DetectAnomaliesRequest = _reflection.GeneratedProtocolMessageType('DetectAnomaliesRequest', (_message.Message,), {
  'DESCRIPTOR' : _DETECTANOMALIESREQUEST,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.DetectAnomaliesRequest)
  })
_sym_db.RegisterMessage(DetectAnomaliesRequest)

DetectAnomalyResult = _reflection.GeneratedProtocolMessageType('DetectAnomalyResult', (_message.Message,), {
  'DESCRIPTOR' : _DETECTANOMALYRESULT,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.DetectAnomalyResult)
  })
_sym_db.RegisterMessage(DetectAnomalyResult)

DetectAnomaliesResponse = _reflection.GeneratedProtocolMessageType('DetectAnomaliesResponse', (_message.Message,), {
  'DESCRIPTOR' : _DETECTANOMALIESRESPONSE,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.DetectAnomaliesResponse)
  })
_sym_db.RegisterMessage(DetectAnomaliesResponse)

ModelDescription = _reflection.GeneratedProtocolMessageType('ModelDescription', (_message.Message,), {
  'DESCRIPTOR' : _MODELDESCRIPTION,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.ModelDescription)
  })
_sym_db.RegisterMessage(ModelDescription)

ModelMetadata = _reflection.GeneratedProtocolMessageType('ModelMetadata', (_message.Message,), {
  'DESCRIPTOR' : _MODELMETADATA,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.ModelMetadata)
  })
_sym_db.RegisterMessage(ModelMetadata)

StartModelRequest = _reflection.GeneratedProtocolMessageType('StartModelRequest', (_message.Message,), {
  'DESCRIPTOR' : _STARTMODELREQUEST,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.StartModelRequest)
  })
_sym_db.RegisterMessage(StartModelRequest)

StartModelResponse = _reflection.GeneratedProtocolMessageType('StartModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _STARTMODELRESPONSE,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.StartModelResponse)
  })
_sym_db.RegisterMessage(StartModelResponse)

StopModelRequest = _reflection.GeneratedProtocolMessageType('StopModelRequest', (_message.Message,), {
  'DESCRIPTOR' : _STOPMODELREQUEST,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.StopModelRequest)
  })
_sym_db.RegisterMessage(StopModelRequest)

StopModelResponse = _reflection.GeneratedProtocolMessageType('StopModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _STOPMODELRESPONSE,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.StopModelResponse)
  })
_sym_db.RegisterMessage(StopModelResponse)

ListModelsRequest = _reflection.GeneratedProtocolMessageType('ListModelsRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTMODELSREQUEST,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.ListModelsRequest)
  })
_sym_db.RegisterMessage(ListModelsRequest)

ListModelsResponse = _reflection.GeneratedProtocolMessageType('ListModelsResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTMODELSRESPONSE,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.ListModelsResponse)
  })
_sym_db.RegisterMessage(ListModelsResponse)

DescribeModelRequest = _reflection.GeneratedProtocolMessageType('DescribeModelRequest', (_message.Message,), {
  'DESCRIPTOR' : _DESCRIBEMODELREQUEST,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.DescribeModelRequest)
  })
_sym_db.RegisterMessage(DescribeModelRequest)

DescribeModelResponse = _reflection.GeneratedProtocolMessageType('DescribeModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _DESCRIBEMODELRESPONSE,
  '__module__' : 'inference_server_pb2'
  # @@protoc_insertion_point(class_scope:AWS.LookoutVision.Edge.DescribeModelResponse)
  })
_sym_db.RegisterMessage(DescribeModelResponse)


DESCRIPTOR._options = None

_INFERENCESERVER = _descriptor.ServiceDescriptor(
  name='InferenceServer',
  full_name='AWS.LookoutVision.Edge.InferenceServer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1404,
  serialized_end=1947,
  methods=[
  _descriptor.MethodDescriptor(
    name='DetectAnomalies',
    full_name='AWS.LookoutVision.Edge.InferenceServer.DetectAnomalies',
    index=0,
    containing_service=None,
    input_type=_DETECTANOMALIESREQUEST,
    output_type=_DETECTANOMALIESRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StartModel',
    full_name='AWS.LookoutVision.Edge.InferenceServer.StartModel',
    index=1,
    containing_service=None,
    input_type=_STARTMODELREQUEST,
    output_type=_STARTMODELRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StopModel',
    full_name='AWS.LookoutVision.Edge.InferenceServer.StopModel',
    index=2,
    containing_service=None,
    input_type=_STOPMODELREQUEST,
    output_type=_STOPMODELRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ListModels',
    full_name='AWS.LookoutVision.Edge.InferenceServer.ListModels',
    index=3,
    containing_service=None,
    input_type=_LISTMODELSREQUEST,
    output_type=_LISTMODELSRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DescribeModel',
    full_name='AWS.LookoutVision.Edge.InferenceServer.DescribeModel',
    index=4,
    containing_service=None,
    input_type=_DESCRIBEMODELREQUEST,
    output_type=_DESCRIBEMODELRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_INFERENCESERVER)

DESCRIPTOR.services_by_name['InferenceServer'] = _INFERENCESERVER

# @@protoc_insertion_point(module_scope)