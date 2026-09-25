import argparse
import pandas as pd
from algo.core.master import InstrumentMaster

def main():
    parser = argparse.ArgumentParser(description="Search the Upstox Instrument Master registry.")
    parser.add_argument("query", type=str, help="The symbol or name to search for (e.g., NATURALGAS)")
    parser.add_argument("-s", "--segment", type=str, help="Filter by segment (e.g., MCX_FO, NSE_EQ)")
    parser.add_argument("-e", "--exchange", type=str, help="Filter by exchange (e.g., MCX, NSE)")
    parser.add_argument("-t", "--type", type=str, help="Filter by instrument type (e.g., FUT, CE, PE, EQ)")
    args = parser.parse_args()

    # Initialize master (this instantly loads the cached Parquet file into memory)
    master = InstrumentMaster()
    
    print(f"Searching for '{args.query}'...\n")
    
    results = []
    query_lower = args.query.lower()
    
    # Fast O(N) scan through the cached dictionary
    for key, meta in master._registry.items():
        name = str(meta.get("name", "")).lower()
        symbol = str(meta.get("tradingsymbol", "")).lower()
        
        if query_lower in name or query_lower in symbol:
            if args.segment and args.segment.lower() not in str(meta.get("segment", "")).lower():
                continue
            if args.exchange and args.exchange.lower() not in str(meta.get("exchange", "")).lower():
                continue
            
            # Check both instrument_type (e.g., OPTIDX) and option_type (e.g., CE, PE)
            if args.type:
                i_type = str(meta.get("instrument_type", "")).lower()
                o_type = str(meta.get("option_type", "")).lower()
                if args.type.lower() not in i_type and args.type.lower() not in o_type:
                    continue
                
            results.append({
                "Key": key,
                "Symbol": meta.get("tradingsymbol", ""),
                "Name": meta.get("name", ""),
                "Type": meta.get("instrument_type", ""),
                "Expiry": meta.get("expiry", ""),
                "Lot Size": meta.get("lot_size", "")
            })

    if not results:
        print("No matching instruments found.")
        return

    # Convert to DataFrame for clean terminal formatting
    df = pd.DataFrame(results)
    
    # Sort chronologically by expiry if available
    if "Expiry" in df.columns:
        df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
        df = df.sort_values(by=["Name", "Expiry_DT"]).drop(columns=["Expiry_DT"])
        
    print(df.to_markdown(index=False, tablefmt="simple"))

if __name__ == "__main__":
    main()
