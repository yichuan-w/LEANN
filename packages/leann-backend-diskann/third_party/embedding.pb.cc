// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: embedding.proto

#include "embedding.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace protoembedding {
class NodeEmbeddingRequestDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<NodeEmbeddingRequest> _instance;
} _NodeEmbeddingRequest_default_instance_;
class NodeEmbeddingResponseDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<NodeEmbeddingResponse> _instance;
} _NodeEmbeddingResponse_default_instance_;
}  // namespace protoembedding
static void InitDefaultsscc_info_NodeEmbeddingRequest_embedding_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::protoembedding::_NodeEmbeddingRequest_default_instance_;
    new (ptr) ::protoembedding::NodeEmbeddingRequest();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::protoembedding::NodeEmbeddingRequest::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_NodeEmbeddingRequest_embedding_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_NodeEmbeddingRequest_embedding_2eproto}, {}};

static void InitDefaultsscc_info_NodeEmbeddingResponse_embedding_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::protoembedding::_NodeEmbeddingResponse_default_instance_;
    new (ptr) ::protoembedding::NodeEmbeddingResponse();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::protoembedding::NodeEmbeddingResponse::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_NodeEmbeddingResponse_embedding_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_NodeEmbeddingResponse_embedding_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_embedding_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_embedding_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_embedding_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_embedding_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::protoembedding::NodeEmbeddingRequest, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::protoembedding::NodeEmbeddingRequest, node_ids_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::protoembedding::NodeEmbeddingResponse, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::protoembedding::NodeEmbeddingResponse, embeddings_data_),
  PROTOBUF_FIELD_OFFSET(::protoembedding::NodeEmbeddingResponse, dimensions_),
  PROTOBUF_FIELD_OFFSET(::protoembedding::NodeEmbeddingResponse, missing_ids_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::protoembedding::NodeEmbeddingRequest)},
  { 6, -1, sizeof(::protoembedding::NodeEmbeddingResponse)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::protoembedding::_NodeEmbeddingRequest_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::protoembedding::_NodeEmbeddingResponse_default_instance_),
};

const char descriptor_table_protodef_embedding_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\017embedding.proto\022\016protoembedding\"(\n\024Nod"
  "eEmbeddingRequest\022\020\n\010node_ids\030\001 \003(\r\"Y\n\025N"
  "odeEmbeddingResponse\022\027\n\017embeddings_data\030"
  "\001 \001(\014\022\022\n\ndimensions\030\002 \003(\005\022\023\n\013missing_ids"
  "\030\003 \003(\rb\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_embedding_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_embedding_2eproto_sccs[2] = {
  &scc_info_NodeEmbeddingRequest_embedding_2eproto.base,
  &scc_info_NodeEmbeddingResponse_embedding_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_embedding_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_embedding_2eproto = {
  false, false, descriptor_table_protodef_embedding_2eproto, "embedding.proto", 174,
  &descriptor_table_embedding_2eproto_once, descriptor_table_embedding_2eproto_sccs, descriptor_table_embedding_2eproto_deps, 2, 0,
  schemas, file_default_instances, TableStruct_embedding_2eproto::offsets,
  file_level_metadata_embedding_2eproto, 2, file_level_enum_descriptors_embedding_2eproto, file_level_service_descriptors_embedding_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_embedding_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_embedding_2eproto)), true);
namespace protoembedding {

// ===================================================================

void NodeEmbeddingRequest::InitAsDefaultInstance() {
}
class NodeEmbeddingRequest::_Internal {
 public:
};

NodeEmbeddingRequest::NodeEmbeddingRequest(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  node_ids_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:protoembedding.NodeEmbeddingRequest)
}
NodeEmbeddingRequest::NodeEmbeddingRequest(const NodeEmbeddingRequest& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      node_ids_(from.node_ids_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:protoembedding.NodeEmbeddingRequest)
}

void NodeEmbeddingRequest::SharedCtor() {
}

NodeEmbeddingRequest::~NodeEmbeddingRequest() {
  // @@protoc_insertion_point(destructor:protoembedding.NodeEmbeddingRequest)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void NodeEmbeddingRequest::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void NodeEmbeddingRequest::ArenaDtor(void* object) {
  NodeEmbeddingRequest* _this = reinterpret_cast< NodeEmbeddingRequest* >(object);
  (void)_this;
}
void NodeEmbeddingRequest::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void NodeEmbeddingRequest::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const NodeEmbeddingRequest& NodeEmbeddingRequest::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_NodeEmbeddingRequest_embedding_2eproto.base);
  return *internal_default_instance();
}


void NodeEmbeddingRequest::Clear() {
// @@protoc_insertion_point(message_clear_start:protoembedding.NodeEmbeddingRequest)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  node_ids_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* NodeEmbeddingRequest::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArena(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // repeated uint32 node_ids = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedUInt32Parser(_internal_mutable_node_ids(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8) {
          _internal_add_node_ids(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* NodeEmbeddingRequest::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:protoembedding.NodeEmbeddingRequest)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated uint32 node_ids = 1;
  {
    int byte_size = _node_ids_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteUInt32Packed(
          1, _internal_node_ids(), byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:protoembedding.NodeEmbeddingRequest)
  return target;
}

size_t NodeEmbeddingRequest::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:protoembedding.NodeEmbeddingRequest)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated uint32 node_ids = 1;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      UInt32Size(this->node_ids_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _node_ids_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void NodeEmbeddingRequest::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:protoembedding.NodeEmbeddingRequest)
  GOOGLE_DCHECK_NE(&from, this);
  const NodeEmbeddingRequest* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<NodeEmbeddingRequest>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:protoembedding.NodeEmbeddingRequest)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:protoembedding.NodeEmbeddingRequest)
    MergeFrom(*source);
  }
}

void NodeEmbeddingRequest::MergeFrom(const NodeEmbeddingRequest& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:protoembedding.NodeEmbeddingRequest)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  node_ids_.MergeFrom(from.node_ids_);
}

void NodeEmbeddingRequest::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:protoembedding.NodeEmbeddingRequest)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void NodeEmbeddingRequest::CopyFrom(const NodeEmbeddingRequest& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:protoembedding.NodeEmbeddingRequest)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool NodeEmbeddingRequest::IsInitialized() const {
  return true;
}

void NodeEmbeddingRequest::InternalSwap(NodeEmbeddingRequest* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  node_ids_.InternalSwap(&other->node_ids_);
}

::PROTOBUF_NAMESPACE_ID::Metadata NodeEmbeddingRequest::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

void NodeEmbeddingResponse::InitAsDefaultInstance() {
}
class NodeEmbeddingResponse::_Internal {
 public:
};

NodeEmbeddingResponse::NodeEmbeddingResponse(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  dimensions_(arena),
  missing_ids_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:protoembedding.NodeEmbeddingResponse)
}
NodeEmbeddingResponse::NodeEmbeddingResponse(const NodeEmbeddingResponse& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      dimensions_(from.dimensions_),
      missing_ids_(from.missing_ids_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  embeddings_data_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_embeddings_data().empty()) {
    embeddings_data_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from._internal_embeddings_data(),
      GetArena());
  }
  // @@protoc_insertion_point(copy_constructor:protoembedding.NodeEmbeddingResponse)
}

void NodeEmbeddingResponse::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_NodeEmbeddingResponse_embedding_2eproto.base);
  embeddings_data_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

NodeEmbeddingResponse::~NodeEmbeddingResponse() {
  // @@protoc_insertion_point(destructor:protoembedding.NodeEmbeddingResponse)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void NodeEmbeddingResponse::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  embeddings_data_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void NodeEmbeddingResponse::ArenaDtor(void* object) {
  NodeEmbeddingResponse* _this = reinterpret_cast< NodeEmbeddingResponse* >(object);
  (void)_this;
}
void NodeEmbeddingResponse::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void NodeEmbeddingResponse::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const NodeEmbeddingResponse& NodeEmbeddingResponse::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_NodeEmbeddingResponse_embedding_2eproto.base);
  return *internal_default_instance();
}


void NodeEmbeddingResponse::Clear() {
// @@protoc_insertion_point(message_clear_start:protoembedding.NodeEmbeddingResponse)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  dimensions_.Clear();
  missing_ids_.Clear();
  embeddings_data_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* NodeEmbeddingResponse::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArena(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // bytes embeddings_data = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_embeddings_data();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int32 dimensions = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_dimensions(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16) {
          _internal_add_dimensions(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated uint32 missing_ids = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedUInt32Parser(_internal_mutable_missing_ids(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24) {
          _internal_add_missing_ids(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* NodeEmbeddingResponse::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:protoembedding.NodeEmbeddingResponse)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // bytes embeddings_data = 1;
  if (this->embeddings_data().size() > 0) {
    target = stream->WriteBytesMaybeAliased(
        1, this->_internal_embeddings_data(), target);
  }

  // repeated int32 dimensions = 2;
  {
    int byte_size = _dimensions_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt32Packed(
          2, _internal_dimensions(), byte_size, target);
    }
  }

  // repeated uint32 missing_ids = 3;
  {
    int byte_size = _missing_ids_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteUInt32Packed(
          3, _internal_missing_ids(), byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:protoembedding.NodeEmbeddingResponse)
  return target;
}

size_t NodeEmbeddingResponse::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:protoembedding.NodeEmbeddingResponse)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int32 dimensions = 2;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int32Size(this->dimensions_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _dimensions_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated uint32 missing_ids = 3;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      UInt32Size(this->missing_ids_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _missing_ids_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // bytes embeddings_data = 1;
  if (this->embeddings_data().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::BytesSize(
        this->_internal_embeddings_data());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void NodeEmbeddingResponse::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:protoembedding.NodeEmbeddingResponse)
  GOOGLE_DCHECK_NE(&from, this);
  const NodeEmbeddingResponse* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<NodeEmbeddingResponse>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:protoembedding.NodeEmbeddingResponse)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:protoembedding.NodeEmbeddingResponse)
    MergeFrom(*source);
  }
}

void NodeEmbeddingResponse::MergeFrom(const NodeEmbeddingResponse& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:protoembedding.NodeEmbeddingResponse)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  dimensions_.MergeFrom(from.dimensions_);
  missing_ids_.MergeFrom(from.missing_ids_);
  if (from.embeddings_data().size() > 0) {
    _internal_set_embeddings_data(from._internal_embeddings_data());
  }
}

void NodeEmbeddingResponse::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:protoembedding.NodeEmbeddingResponse)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void NodeEmbeddingResponse::CopyFrom(const NodeEmbeddingResponse& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:protoembedding.NodeEmbeddingResponse)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool NodeEmbeddingResponse::IsInitialized() const {
  return true;
}

void NodeEmbeddingResponse::InternalSwap(NodeEmbeddingResponse* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  dimensions_.InternalSwap(&other->dimensions_);
  missing_ids_.InternalSwap(&other->missing_ids_);
  embeddings_data_.Swap(&other->embeddings_data_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}

::PROTOBUF_NAMESPACE_ID::Metadata NodeEmbeddingResponse::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace protoembedding
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::protoembedding::NodeEmbeddingRequest* Arena::CreateMaybeMessage< ::protoembedding::NodeEmbeddingRequest >(Arena* arena) {
  return Arena::CreateMessageInternal< ::protoembedding::NodeEmbeddingRequest >(arena);
}
template<> PROTOBUF_NOINLINE ::protoembedding::NodeEmbeddingResponse* Arena::CreateMaybeMessage< ::protoembedding::NodeEmbeddingResponse >(Arena* arena) {
  return Arena::CreateMessageInternal< ::protoembedding::NodeEmbeddingResponse >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
