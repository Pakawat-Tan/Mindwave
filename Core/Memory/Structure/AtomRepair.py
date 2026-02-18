"""
ATOM Binary Format Repair Module

This module provides functionality to repair corrupted ATOM binary files.
It can detect and fix various types of corruption including:
- Corrupted magic bytes
- Invalid or missing CRC32
- Truncated data
- Size mismatches
- Header corruption
"""

import struct
import zlib
from dataclasses import dataclass
from typing import Optional, List
from .AtomStructure import (
    AtomData,
    AtomHeader,
    AtomBinaryFormat,
    MAGIC,
    VERSION,
    HEADER_SIZE,
    FOOTER_SIZE,
)

@dataclass
class RepairReport:
    """Report of repair operations performed"""
    success: bool
    original_size: int
    repaired_size: int
    issues_found: List[str]
    fixes_applied: List[str]
    warnings: List[str]
    recovered_data: Optional[AtomData] = None
    
    def __str__(self):
        lines = []
        lines.append("=" * 60)
        lines.append("ATOM Binary Format Repair Report")
        lines.append("=" * 60)
        lines.append(f"Status: {'✓ SUCCESS' if self.success else '✗ FAILED'}")
        lines.append(f"Original Size: {self.original_size} bytes")
        lines.append(f"Repaired Size: {self.repaired_size} bytes")
        
        if self.issues_found:
            lines.append(f"\nIssues Found ({len(self.issues_found)}):")
            for issue in self.issues_found:
                lines.append(f"  ✗ {issue}")
        
        if self.fixes_applied:
            lines.append(f"\nFixes Applied ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                lines.append(f"  ✓ {fix}")
        
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        if self.recovered_data:
            lines.append(f"\nRecovered Data:")
            lines.append(f"  Payload: {len(self.recovered_data.payload)} bytes")
            lines.append(f"  Metadata: {len(self.recovered_data.metadata)} bytes")
            lines.append(f"  Source: {len(self.recovered_data.source)} bytes")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class AtomRepair:
    """Repair utility for corrupted ATOM binary files"""
    
    @staticmethod
    def diagnose(binary_data: bytes) -> List[str]:
        """
        Diagnose issues with ATOM binary data without attempting repair
        
        Returns:
            List of issues found
        """
        issues = []
        
        # Check minimum size
        if len(binary_data) < HEADER_SIZE + FOOTER_SIZE:
            issues.append(f"File too small: {len(binary_data)} bytes (minimum: {HEADER_SIZE + FOOTER_SIZE})")
            return issues
        
        # Check magic bytes
        if binary_data[:4] != MAGIC:
            issues.append(f"Invalid magic bytes: {binary_data[:4]} (expected: {MAGIC})")
        
        # Try to parse header
        try:
            header = AtomHeader.from_bytes(binary_data[:HEADER_SIZE])
            
            # Check version
            if header.version != VERSION:
                issues.append(f"Unsupported version: {header.version} (current: {VERSION})")
            
            # Check reserved field
            if header.reserved != 0:
                issues.append(f"Reserved field not zero: {header.reserved}")
            
            # Check size consistency
            expected_size = HEADER_SIZE + header.payload_len + header.metadata_len + header.source_len + FOOTER_SIZE
            if len(binary_data) != expected_size:
                issues.append(f"Size mismatch: {len(binary_data)} != {expected_size}")
            
            # Check CRC32
            if len(binary_data) >= expected_size:
                crc_offset = expected_size - FOOTER_SIZE
                stored_crc = struct.unpack('>I', binary_data[crc_offset:crc_offset + FOOTER_SIZE])[0]
                calculated_crc = zlib.crc32(binary_data[:crc_offset]) & 0xffffffff
                
                if stored_crc != calculated_crc:
                    issues.append(f"CRC32 mismatch: {stored_crc:08x} != {calculated_crc:08x}")
        
        except Exception as e:
            issues.append(f"Header parsing error: {e}")
        
        return issues
    
    @staticmethod
    def repair(binary_data: bytes, aggressive: bool = False) -> RepairReport:
        """
        Attempt to repair corrupted ATOM binary data
        
        Args:
            binary_data: Corrupted binary data
            aggressive: If True, attempt more aggressive repair strategies
            
        Returns:
            RepairReport with details of repair attempt
        """
        issues = []
        fixes = []
        warnings = []
        original_size = len(binary_data)
        
        # Diagnose first
        issues = AtomRepair.diagnose(binary_data)
        
        if not issues:
            # No issues, return original data
            try:
                data = AtomBinaryFormat.decode(binary_data)
                return RepairReport(
                    success=True,
                    original_size=original_size,
                    repaired_size=original_size,
                    issues_found=[],
                    fixes_applied=["No repair needed - file is valid"],
                    warnings=[],
                    recovered_data=data
                )
            except Exception as e:
                issues.append(f"Unexpected decode error: {e}")
        
        # Start repair process
        repaired_data = bytearray(binary_data)
        
        # Fix 1: Repair magic bytes
        if repaired_data[:4] != MAGIC:
            repaired_data[:4] = MAGIC
            fixes.append("Repaired magic bytes to 'ATOM'")
        
        # Fix 2: Try to extract header
        try:
            header = AtomHeader.from_bytes(bytes(repaired_data[:HEADER_SIZE]))
        except Exception as e:
            if aggressive:
                # Aggressive: Try to reconstruct header from partial data
                fixes.append(f"Attempted header reconstruction (aggressive mode)")
                return AtomRepair._aggressive_repair(binary_data, issues, fixes, warnings)
            else:
                return RepairReport(
                    success=False,
                    original_size=original_size,
                    repaired_size=len(repaired_data),
                    issues_found=issues,
                    fixes_applied=fixes,
                    warnings=[f"Header too corrupted. Try aggressive mode."],
                    recovered_data=None
                )
        
        # Fix 3: Repair reserved field
        if header.reserved != 0:
            struct.pack_into('>H', repaired_data, 6, 0)
            fixes.append("Reset reserved field to 0")
            header.reserved = 0
        
        # Fix 4: Handle size mismatches
        expected_size = HEADER_SIZE + header.payload_len + header.metadata_len + header.source_len + FOOTER_SIZE
        
        if len(repaired_data) < expected_size:
            # Truncated file
            if aggressive:
                # Try to salvage what we can
                fixes.append("File truncated - attempting partial recovery")
                return AtomRepair._recover_truncated(bytes(repaired_data), header, issues, fixes, warnings)
            else:
                warnings.append(f"File truncated: {len(repaired_data)} < {expected_size}. Try aggressive mode.")
                return RepairReport(
                    success=False,
                    original_size=original_size,
                    repaired_size=len(repaired_data),
                    issues_found=issues,
                    fixes_applied=fixes,
                    warnings=warnings,
                    recovered_data=None
                )
        
        elif len(repaired_data) > expected_size:
            # Extra data at end
            repaired_data = repaired_data[:expected_size]
            fixes.append(f"Truncated {len(binary_data) - expected_size} extra bytes at end")
        
        # Fix 5: Recalculate and fix CRC32
        crc_offset = expected_size - FOOTER_SIZE
        calculated_crc = zlib.crc32(repaired_data[:crc_offset]) & 0xffffffff
        struct.pack_into('>I', repaired_data, crc_offset, calculated_crc)
        fixes.append(f"Recalculated CRC32: {calculated_crc:08x}")
        
        # Try to decode repaired data
        try:
            recovered_data = AtomBinaryFormat.decode(bytes(repaired_data))
            
            # Validate recovered data makes sense
            if len(recovered_data.payload) == 0 and len(recovered_data.metadata) == 0:
                warnings.append("Recovered data appears empty")
            
            return RepairReport(
                success=True,
                original_size=original_size,
                repaired_size=len(repaired_data),
                issues_found=issues,
                fixes_applied=fixes,
                warnings=warnings,
                recovered_data=recovered_data
            )
        
        except Exception as e:
            return RepairReport(
                success=False,
                original_size=original_size,
                repaired_size=len(repaired_data),
                issues_found=issues,
                fixes_applied=fixes,
                warnings=[f"Repair failed: {e}"],
                recovered_data=None
            )
    
    @staticmethod
    def _recover_truncated(binary_data: bytes, header: AtomHeader, 
                          issues: List[str], fixes: List[str], 
                          warnings: List[str]) -> RepairReport:
        """Attempt to recover data from truncated file"""
        available_body = len(binary_data) - HEADER_SIZE
        
        # Calculate what we can recover
        payload_size = min(header.payload_len, available_body)
        remaining = available_body - payload_size
        
        metadata_size = min(header.metadata_len, remaining)
        remaining -= metadata_size
        
        source_size = min(header.source_len, remaining)
        
        # Extract what we have
        offset = HEADER_SIZE
        payload = binary_data[offset:offset + payload_size]
        offset += payload_size
        
        metadata = binary_data[offset:offset + metadata_size] if metadata_size > 0 else b''
        offset += metadata_size
        
        source = binary_data[offset:offset + source_size] if source_size > 0 else b''
        
        warnings.append(f"Recovered {payload_size}/{header.payload_len} bytes of payload")
        if metadata_size < header.metadata_len:
            warnings.append(f"Lost {header.metadata_len - metadata_size} bytes of metadata")
        if source_size < header.source_len:
            warnings.append(f"Lost {header.source_len - source_size} bytes of source")
        
        recovered_data = AtomData(
            payload=payload,
            metadata=metadata,
            source=source,
            flags=header.flags,
            created_ts_ms=header.created_ts_ms
        )
        
        fixes.append("Partial recovery from truncated file")
        
        return RepairReport(
            success=True,
            original_size=len(binary_data),
            repaired_size=len(binary_data),
            issues_found=issues,
            fixes_applied=fixes,
            warnings=warnings,
            recovered_data=recovered_data
        )
    
    @staticmethod
    def _aggressive_repair(binary_data: bytes, issues: List[str], 
                          fixes: List[str], warnings: List[str]) -> RepairReport:
        """Aggressive repair - try to find ATOM structure anywhere in file"""
        
        # Search for magic bytes
        magic_positions = []
        for i in range(len(binary_data) - 4):
            if binary_data[i:i+4] == MAGIC:
                magic_positions.append(i)
        
        if not magic_positions:
            warnings.append("No ATOM magic bytes found anywhere in file")
            return RepairReport(
                success=False,
                original_size=len(binary_data),
                repaired_size=0,
                issues_found=issues,
                fixes_applied=fixes,
                warnings=warnings,
                recovered_data=None
            )
        
        fixes.append(f"Found {len(magic_positions)} potential ATOM structure(s)")
        
        # Try each position
        for pos in magic_positions:
            try:
                # Try to parse from this position
                if pos + HEADER_SIZE <= len(binary_data):
                    candidate = binary_data[pos:]
                    report = AtomRepair.repair(candidate, aggressive=False)
                    
                    if report.success:
                        fixes.append(f"Successfully recovered from offset {pos}")
                        report.fixes_applied = fixes + report.fixes_applied
                        report.issues_found = issues
                        return report
            except:
                continue
        
        warnings.append("Could not recover valid ATOM structure from any position")
        return RepairReport(
            success=False,
            original_size=len(binary_data),
            repaired_size=0,
            issues_found=issues,
            fixes_applied=fixes,
            warnings=warnings,
            recovered_data=None
        )
    
    @staticmethod
    def repair_file(input_path: str, output_path: str = None, 
                    aggressive: bool = False) -> RepairReport:
        """
        Repair a corrupted ATOM file
        
        Args:
            input_path: Path to corrupted file
            output_path: Path to save repaired file (optional)
            aggressive: Use aggressive repair strategies
            
        Returns:
            RepairReport
        """
        # Read corrupted file
        with open(input_path, 'rb') as f:
            binary_data = f.read()
        
        # Attempt repair
        report = AtomRepair.repair(binary_data, aggressive=aggressive)
        
        # Save repaired file if successful
        if report.success and output_path:
            repaired_binary = AtomBinaryFormat.encode(report.recovered_data)
            with open(output_path, 'wb') as f:
                f.write(repaired_binary)
            report.fixes_applied.append(f"Saved repaired file to {output_path}")
        
        return report


def quick_check(filepath: str) -> bool:
    """
    Quick check if ATOM file is valid
    
    Returns:
        True if valid, False if corrupted
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        AtomBinaryFormat.decode(data)
        return True
    except:
        return False


def auto_repair(filepath: str, backup: bool = True) -> bool:
    """
    Automatically repair ATOM file in place
    
    Args:
        filepath: Path to file
        backup: Create .bak backup before repair
        
    Returns:
        True if repair successful
    """
    # Create backup if requested
    if backup:
        import shutil
        backup_path = filepath + '.bak'
        shutil.copy2(filepath, backup_path)
    
    # Try repair
    report = AtomRepair.repair_file(filepath, filepath + '.repaired', aggressive=True)
    
    if report.success:
        # Replace original with repaired
        import os
        os.replace(filepath + '.repaired', filepath)
        return True
    else:
        # Clean up failed repair
        import os
        if os.path.exists(filepath + '.repaired'):
            os.remove(filepath + '.repaired')
        return False