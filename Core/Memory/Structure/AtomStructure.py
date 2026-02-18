"""
ATOM Binary Format Handler

Binary Structure:
[ Header ]
├─ magic (4)            : b"ATOM"
├─ version (1)          : uint8
├─ flags (1)            : uint8
├─ reserved (2)         : uint16 (must be 0)
├─ created_ts_ms (8)    : int64 (epoch ms)
├─ payload_len (4)      : uint32
├─ metadata_len (4)     : uint32
├─ source_len (4)       : uint32

[ Body ]
├─ payload_bytes
├─ metadata_bytes
├─ source_bytes

[ Footer ]
└─ crc32 (4)            : uint32 (header + body)
"""

import struct
import time
import zlib
from dataclasses import dataclass
from typing import Optional

# Constants
MAGIC = b"ATOM"
HEADER_SIZE = 28  # 4 + 1 + 1 + 2 + 8 + 4 + 4 + 4 = 28 bytes
FOOTER_SIZE = 4
VERSION = 1


@dataclass
class AtomHeader:
    """ATOM Binary Format Header"""
    magic: bytes = MAGIC
    version: int = VERSION
    flags: int = 0
    reserved: int = 0
    created_ts_ms: int = 0
    payload_len: int = 0
    metadata_len: int = 0
    source_len: int = 0
    
    def to_bytes(self) -> bytes:
        """Convert header to bytes"""
        return struct.pack(
            '>4sBBHqIII',
            self.magic,
            self.version,
            self.flags,
            self.reserved,
            self.created_ts_ms,
            self.payload_len,
            self.metadata_len,
            self.source_len
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'AtomHeader':
        """Parse header from bytes"""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Header data too short: {len(data)} < {HEADER_SIZE}")
        
        unpacked = struct.unpack('>4sBBHqIII', data[:HEADER_SIZE])
        
        magic, version, flags, reserved, created_ts_ms, payload_len, metadata_len, source_len = unpacked
        
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic} != {MAGIC}")
        
        return cls(
            magic=magic,
            version=version,
            flags=flags,
            reserved=reserved,
            created_ts_ms=created_ts_ms,
            payload_len=payload_len,
            metadata_len=metadata_len,
            source_len=source_len
        )

@dataclass
class AtomData:
    """ATOM Binary Format Data Container"""
    payload: bytes
    metadata: bytes = b''
    source: bytes = b''
    flags: int = 0
    created_ts_ms: Optional[int] = None
    
    def __post_init__(self):
        if self.created_ts_ms is None:
            self.created_ts_ms = int(time.time() * 1000)


class AtomBinaryFormat:
    """Handler for ATOM Binary Format"""
    
    @staticmethod
    def encode(data: AtomData) -> bytes:
        """
        Encode AtomData into ATOM binary format
        
        Args:
            data: AtomData object containing payload, metadata, and source
            
        Returns:
            Complete binary data with header, body, and footer
        """
        # Create header
        header = AtomHeader(
            version=VERSION,
            flags=data.flags,
            reserved=0,
            created_ts_ms=data.created_ts_ms,
            payload_len=len(data.payload),
            metadata_len=len(data.metadata),
            source_len=len(data.source)
        )
        
        # Build header bytes
        header_bytes = header.to_bytes()
        
        # Build body bytes
        body_bytes = data.payload + data.metadata + data.source
        
        # Calculate CRC32 (header + body)
        crc_data = header_bytes + body_bytes
        crc32 = zlib.crc32(crc_data) & 0xffffffff
        
        # Build footer
        footer_bytes = struct.pack('>I', crc32)
        
        # Combine all parts
        return header_bytes + body_bytes + footer_bytes
    
    @staticmethod
    def decode(binary_data: bytes) -> AtomData:
        """
        Decode ATOM binary format into AtomData
        
        Args:
            binary_data: Complete binary data with header, body, and footer
            
        Returns:
            AtomData object
            
        Raises:
            ValueError: If data is corrupted or invalid
        """
        if len(binary_data) < HEADER_SIZE + FOOTER_SIZE:
            raise ValueError("Binary data too short")
        
        # Parse header
        header = AtomHeader.from_bytes(binary_data[:HEADER_SIZE])
        
        # Calculate expected total size
        expected_size = HEADER_SIZE + header.payload_len + header.metadata_len + header.source_len + FOOTER_SIZE
        
        if len(binary_data) != expected_size:
            raise ValueError(f"Size mismatch: {len(binary_data)} != {expected_size}")
        
        # Extract body parts
        offset = HEADER_SIZE
        payload = binary_data[offset:offset + header.payload_len]
        offset += header.payload_len
        
        metadata = binary_data[offset:offset + header.metadata_len]
        offset += header.metadata_len
        
        source = binary_data[offset:offset + header.source_len]
        offset += header.source_len
        
        # Extract and verify CRC32
        footer_data = binary_data[offset:offset + FOOTER_SIZE]
        stored_crc32 = struct.unpack('>I', footer_data)[0]
        
        # Calculate CRC32 of header + body
        crc_data = binary_data[:offset]
        calculated_crc32 = zlib.crc32(crc_data) & 0xffffffff
        
        if stored_crc32 != calculated_crc32:
            raise ValueError(f"CRC32 mismatch: {stored_crc32:08x} != {calculated_crc32:08x}")
        
        return AtomData(
            payload=payload,
            metadata=metadata,
            source=source,
            flags=header.flags,
            created_ts_ms=header.created_ts_ms
        )
    
    @staticmethod
    def save(filepath: str, data: AtomData) -> None:
        """Save AtomData to file"""
        binary_data = AtomBinaryFormat.encode(data)
        with open(filepath, 'wb') as f:
            f.write(binary_data)
    
    @staticmethod
    def load(filepath: str) -> AtomData:
        """Load AtomData from file"""
        with open(filepath, 'rb') as f:
            binary_data = f.read()
        return AtomBinaryFormat.decode(binary_data)