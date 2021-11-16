# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. // SPDX-License-Identifier: MIT-0
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import inference_server_pb2 as inference__server__pb2


class InferenceServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DetectAnomalies = channel.unary_unary(
                '/AWS.LookoutVision.Edge.InferenceServer/DetectAnomalies',
                request_serializer=inference__server__pb2.DetectAnomaliesRequest.SerializeToString,
                response_deserializer=inference__server__pb2.DetectAnomaliesResponse.FromString,
                )
        self.StartModel = channel.unary_unary(
                '/AWS.LookoutVision.Edge.InferenceServer/StartModel',
                request_serializer=inference__server__pb2.StartModelRequest.SerializeToString,
                response_deserializer=inference__server__pb2.StartModelResponse.FromString,
                )
        self.StopModel = channel.unary_unary(
                '/AWS.LookoutVision.Edge.InferenceServer/StopModel',
                request_serializer=inference__server__pb2.StopModelRequest.SerializeToString,
                response_deserializer=inference__server__pb2.StopModelResponse.FromString,
                )
        self.ListModels = channel.unary_unary(
                '/AWS.LookoutVision.Edge.InferenceServer/ListModels',
                request_serializer=inference__server__pb2.ListModelsRequest.SerializeToString,
                response_deserializer=inference__server__pb2.ListModelsResponse.FromString,
                )
        self.DescribeModel = channel.unary_unary(
                '/AWS.LookoutVision.Edge.InferenceServer/DescribeModel',
                request_serializer=inference__server__pb2.DescribeModelRequest.SerializeToString,
                response_deserializer=inference__server__pb2.DescribeModelResponse.FromString,
                )


class InferenceServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def DetectAnomalies(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StartModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StopModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListModels(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DescribeModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DetectAnomalies': grpc.unary_unary_rpc_method_handler(
                    servicer.DetectAnomalies,
                    request_deserializer=inference__server__pb2.DetectAnomaliesRequest.FromString,
                    response_serializer=inference__server__pb2.DetectAnomaliesResponse.SerializeToString,
            ),
            'StartModel': grpc.unary_unary_rpc_method_handler(
                    servicer.StartModel,
                    request_deserializer=inference__server__pb2.StartModelRequest.FromString,
                    response_serializer=inference__server__pb2.StartModelResponse.SerializeToString,
            ),
            'StopModel': grpc.unary_unary_rpc_method_handler(
                    servicer.StopModel,
                    request_deserializer=inference__server__pb2.StopModelRequest.FromString,
                    response_serializer=inference__server__pb2.StopModelResponse.SerializeToString,
            ),
            'ListModels': grpc.unary_unary_rpc_method_handler(
                    servicer.ListModels,
                    request_deserializer=inference__server__pb2.ListModelsRequest.FromString,
                    response_serializer=inference__server__pb2.ListModelsResponse.SerializeToString,
            ),
            'DescribeModel': grpc.unary_unary_rpc_method_handler(
                    servicer.DescribeModel,
                    request_deserializer=inference__server__pb2.DescribeModelRequest.FromString,
                    response_serializer=inference__server__pb2.DescribeModelResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'AWS.LookoutVision.Edge.InferenceServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class InferenceServer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def DetectAnomalies(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AWS.LookoutVision.Edge.InferenceServer/DetectAnomalies',
            inference__server__pb2.DetectAnomaliesRequest.SerializeToString,
            inference__server__pb2.DetectAnomaliesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StartModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AWS.LookoutVision.Edge.InferenceServer/StartModel',
            inference__server__pb2.StartModelRequest.SerializeToString,
            inference__server__pb2.StartModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StopModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AWS.LookoutVision.Edge.InferenceServer/StopModel',
            inference__server__pb2.StopModelRequest.SerializeToString,
            inference__server__pb2.StopModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListModels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AWS.LookoutVision.Edge.InferenceServer/ListModels',
            inference__server__pb2.ListModelsRequest.SerializeToString,
            inference__server__pb2.ListModelsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DescribeModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/AWS.LookoutVision.Edge.InferenceServer/DescribeModel',
            inference__server__pb2.DescribeModelRequest.SerializeToString,
            inference__server__pb2.DescribeModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
