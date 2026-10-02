// NVENC SDK 13 bridge. No sleeps, frame pacing, deadline drops, or bitrate pacer.
// frameRateNum describes MEDIA time for CBR; it does not throttle EncodeFrame.
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <time.h>
#include <unistd.h>
#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "Utils/NvCodecUtils.h"
#include "shared_input.h"

static uint64_t ns() {
    timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;
}
static void require(bool b, const char* msg) { if (!b) throw std::runtime_error(msg); }
static bool read_full(void* p, size_t n, bool eof_ok=false) {
    size_t i=0;
    while (i<n) {
        size_t k=fread(static_cast<char*>(p)+i,1,n-i,stdin);
        if (!k) {
            if (i==0 && eof_ok && feof(stdin)) return false;
            throw std::runtime_error("truncated input record");
        }
        i+=k;
    }
    return true;
}
static void put(FILE* f,const void* p,size_t n) {
    require(fwrite(p,1,n,f)==n,"binary output write failed");
}
template<class T> static void put(FILE* f,T v) { put(f,&v,sizeof(v)); }
template<class T> static void set709(T& v) {
    v.videoSignalTypePresentFlag=1;
    v.videoFormat=static_cast<decltype(v.videoFormat)>(5);
    v.videoFullRangeFlag=0; v.colourDescriptionPresentFlag=1;
    v.colourPrimaries=static_cast<decltype(v.colourPrimaries)>(1);
    v.transferCharacteristics=static_cast<decltype(v.transferCharacteristics)>(1);
    v.colourMatrix=static_cast<decltype(v.colourMatrix)>(1);
}

int main(int argc,char** argv) {
    FILE* wire=fdopen(dup(STDOUT_FILENO),"wb");
    if (!wire || dup2(STDERR_FILENO,STDOUT_FILENO)<0) return 1;
    try {
        require(argc==4,"usage: nvenc_uncapped {h264|hevc|av1} BITRATE_BPS SHARED_FD");
        const std::string codec=argv[1];
        const int W=4096,H=2048,fps=60;
        const uint32_t bitrate=std::stoul(argv[2]);
        require(codec=="h264" || codec=="hevc" || codec=="av1","unknown codec");
        const GUID guid=codec=="h264" ? NV_ENC_CODEC_H264_GUID :
                        codec=="hevc" ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_AV1_GUID;
        const int block=codec=="h264" ? 16 : codec=="hevc" ? 32 : 64;
        require(cuInit(0)==CUDA_SUCCESS,"cuInit failed");
        CUdevice dev{}; CUcontext ctx{};
        require(cuDeviceGet(&dev,0)==CUDA_SUCCESS,"cuDeviceGet failed");
        require(cuCtxCreate_v2(&ctx,0,dev)==CUDA_SUCCESS,"cuCtxCreate failed");
        {
            NvEncoderCuda enc(ctx,W,H,NV_ENC_BUFFER_FORMAT_NV12,0);
            NV_ENC_INITIALIZE_PARAMS init{NV_ENC_INITIALIZE_PARAMS_VER};
            NV_ENC_CONFIG cfg{NV_ENC_CONFIG_VER};
            init.encodeConfig=&cfg;
            enc.CreateDefaultEncoderParams(&init,guid,NV_ENC_PRESET_P7_GUID,
                                           NV_ENC_TUNING_INFO_LOW_LATENCY);
            init.frameRateNum=fps; init.frameRateDen=1;
            init.enableEncodeAsync=0; init.enablePTD=1;
            cfg.gopLength=240; cfg.frameIntervalP=1;
            cfg.rcParams.rateControlMode=NV_ENC_PARAMS_RC_CBR;
            cfg.rcParams.averageBitRate=bitrate; cfg.rcParams.maxBitRate=bitrate;
            cfg.rcParams.vbvBufferSize=bitrate/4; cfg.rcParams.vbvInitialDelay=bitrate/4;
            cfg.rcParams.multiPass=NV_ENC_MULTI_PASS_DISABLED;
            cfg.rcParams.enableAQ=0; cfg.rcParams.enableTemporalAQ=0;
            cfg.rcParams.enableLookahead=0; cfg.rcParams.lookaheadDepth=0;
            cfg.rcParams.zeroReorderDelay=1; cfg.rcParams.qpMapMode=NV_ENC_QP_MAP_DELTA;
            if (codec=="h264") {
                cfg.profileGUID=NV_ENC_H264_PROFILE_MAIN_GUID;
                auto& c=cfg.encodeCodecConfig.h264Config;
                c.level=NV_ENC_LEVEL_H264_52; c.idrPeriod=240;
                c.repeatSPSPPS=1; c.outputAUD=1; set709(c.h264VUIParameters);
            } else if (codec=="hevc") {
                cfg.profileGUID=NV_ENC_HEVC_PROFILE_MAIN_GUID;
                auto& c=cfg.encodeCodecConfig.hevcConfig;
                c.idrPeriod=240; c.repeatSPSPPS=1; c.outputAUD=1;
                set709(c.hevcVUIParameters);
            } else {
                cfg.profileGUID=NV_ENC_AV1_PROFILE_MAIN_GUID;
                auto& c=cfg.encodeCodecConfig.av1Config;
                c.idrPeriod=240; c.repeatSeqHdr=1;
                c.colorPrimaries=static_cast<decltype(c.colorPrimaries)>(1);
                c.transferCharacteristics=static_cast<decltype(c.transferCharacteristics)>(1);
                c.matrixCoefficients=static_cast<decltype(c.matrixCoefficients)>(1);
                c.colorRange=0;
            }
            enc.CreateEncoder(&init);  // Unsupported settings cause a recorded error, never fallback.
            const size_t raw_bytes=size_t(W)*H*3/2;
            SharedInput shared(std::stoi(argv[3]),raw_bytes);
            bool pinned=cuMemHostRegister(shared.data,raw_bytes,CU_MEMHOSTREGISTER_PORTABLE)==CUDA_SUCCESS;
            std::cerr << "codec=" << codec << " preset=p7 tuning=low_latency CBR=" << bitrate
                      << " media_fps=60 clock_pacing=none QP_block=" << block
                      << " host_memory=" << (pinned ? "pinned" : "pageable") << std::endl;
            std::vector<int8_t> qp(size_t((W+block-1)/block)*((H+block-1)/block));
            uint64_t expected=0;
            put(wire,"RDY1",4); fflush(wire);
            for (;;) {
                char magic[4];
                if (!read_full(magic,4,true)) break;
                uint64_t id; uint32_t raw_size,qp_size;
                require(!memcmp(magic,"FRM1",4),"bad input magic");
                read_full(&id,8); read_full(&raw_size,4); read_full(&qp_size,4);
                require(id==expected,"input source IDs must be contiguous");
                require(raw_size==raw_bytes && qp_size==qp.size(),"wrong input geometry");
                read_full(qp.data(),qp.size());
                for (auto v:qp) require(v>=-3 && v<=2,"unexpected allocation QP delta");
                uint64_t begin=ns();
                const auto* f=enc.GetNextInputFrame();
                NvEncoderCuda::CopyToDeviceFrame(ctx,shared.data,W,(CUdeviceptr)f->inputPtr,
                    f->pitch,W,H,CU_MEMORYTYPE_HOST,f->bufferFormat,f->chromaOffsets,
                    f->numChromaPlanes,false,0);
                NV_ENC_PIC_PARAMS pic{NV_ENC_PIC_PARAMS_VER};
                pic.pictureStruct=NV_ENC_PIC_STRUCT_FRAME;
                pic.inputTimeStamp=id; pic.inputDuration=1;
                pic.qpDeltaMap=qp.data(); pic.qpDeltaMapSize=uint32_t(qp.size());
                if (id%240==0) {
                    pic.encodePicFlags=NV_ENC_PIC_FLAG_FORCEIDR;
                    if (codec!="av1") pic.encodePicFlags|=NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
                }
                std::vector<NvEncOutputFrame> packets;
                enc.EncodeFrame(packets,&pic);
                uint64_t end=ns();
                require(packets.size()==1 && !packets[0].frame.empty(),
                        "encoder did not return exactly one completed output frame");
                const auto& packet=packets[0].frame;
                require(packet.size()<20000000,"unreasonable output size");
                put(wire,"AU01",4); put(wire,id); put(wire,begin); put(wire,end);
                put(wire,uint32_t(packet.size())); put(wire,packet.data(),packet.size());
                require(fflush(wire)==0,"output flush failed");
                ++expected;
            }
            std::vector<NvEncOutputFrame> tail;
            enc.EndEncode(tail);
            for (const auto& p:tail) require(p.frame.empty(),"unexpected delayed frame at EOS");
            enc.DestroyEncoder();
            if (pinned) require(cuMemHostUnregister(shared.data)==CUDA_SUCCESS,"host unregister failed");
        }
        cuCtxDestroy(ctx); fclose(wire); return 0;
    } catch (const std::exception& e) {
        std::cerr << "NVENC ERROR: " << e.what() << std::endl;
        fclose(wire); return 1;
    }
}
