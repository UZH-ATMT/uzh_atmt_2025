import torch
import torch.nn.functional as F
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel
from seq2seq.beam import BeamSearch, BeamSearchNode

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 1):
    """Decodes a sequence without teacher forcing. 
    
    Args:
        beam_size: Size of beam for beam search. If beam_size=1, uses greedy decoding.
    """
    if beam_size <= 1:
        return _greedy_decode(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, device)
    else:
        return _beam_search_decode(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, device, beam_size)


def _greedy_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                   tgt_tokenizer: spm.SentencePieceProcessor, device: torch.device):
    """Greedy decoding (original implementation)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        # Forward pass: use only the generated tokens so far
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens


def _beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, 
                        max_out_len: int, tgt_tokenizer: spm.SentencePieceProcessor, device: torch.device, 
                        beam_size: int):
    """Beam search decoding"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    
    # Process each sentence in the batch independently
    all_predictions = []
    for batch_idx in range(batch_size):
        # Get single sentence
        src = src_tokens[batch_idx:batch_idx+1]  # (1, src_len)
        src_mask = src_pad_mask[batch_idx:batch_idx+1] if src_pad_mask is not None else None
        
        # Initialize beam search
        beam_search = BeamSearch(beam_size=beam_size, max_len=max_out_len, pad=PAD)
        
        # Start with BOS token
        start_sequence = torch.tensor([[BOS]], dtype=torch.long, device=device)
        start_node = BeamSearchNode(
            search=beam_search,
            emb=None,
            lstm_out=None,
            final_hidden=None,
            final_cell=None,
            mask=None,
            sequence=start_sequence[0],
            logProb=0.0,
            length=1
        )
        beam_search.add(score=0.0, node=start_node)
        
        # Beam search loop
        for step in range(max_out_len):
            # Get current beams
            current_beams = beam_search.get_current_beams()
            if len(current_beams) == 0:
                break
            
            # Prepare batch of current sequences
            sequences = []
            for score, node in current_beams:
                sequences.append(node.sequence)
            
            # Pad sequences to same length
            max_len_beam = max(len(seq) for seq in sequences)
            padded_sequences = []
            for seq in sequences:
                if len(seq) < max_len_beam:
                    padding = torch.full((max_len_beam - len(seq),), PAD, dtype=torch.long, device=device)
                    padded_seq = torch.cat([seq, padding])
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq)
            
            batch_sequences = torch.stack(padded_sequences).unsqueeze(0)  # (1, beam_size, seq_len)
            batch_sequences = batch_sequences.squeeze(0)  # (beam_size, seq_len)
            
            # Expand source for beam
            src_expanded = src.expand(len(current_beams), -1)  # (beam_size, src_len)
            src_mask_expanded = src_mask.expand(len(current_beams), -1, -1, -1) if src_mask is not None else None
            
            # Create target mask
            max_decoder_len = model.decoder.pos_embed.size(1)
            if batch_sequences.size(1) > max_decoder_len:
                batch_sequences = batch_sequences[:, :max_decoder_len]
            trg_mask = (batch_sequences == PAD).unsqueeze(1).unsqueeze(2)
            
            # Forward pass
            with torch.no_grad():
                output = model(src_expanded, src_mask_expanded, batch_sequences, trg_mask)
                # Get logits for last position
                logits = output[:, -1, :]  # (beam_size, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1)
            
            # Expand each beam
            for beam_idx, (score, node) in enumerate(current_beams):
                # Get top-k tokens for this beam
                topk_log_probs, topk_indices = log_probs[beam_idx].topk(beam_size)
                
                for k in range(beam_size):
                    token = topk_indices[k].item()
                    token_log_prob = topk_log_probs[k].item()
                    
                    # Create new sequence
                    new_sequence = torch.cat([node.sequence, torch.tensor([token], device=device)])
                    new_log_prob = node.logp + token_log_prob
                    new_length = node.length + 1
                    
                    # Create new node
                    new_node = BeamSearchNode(
                        search=beam_search,
                        emb=None,
                        lstm_out=None,
                        final_hidden=None,
                        final_cell=None,
                        mask=None,
                        sequence=new_sequence,
                        logProb=new_log_prob,
                        length=new_length
                    )
                    
                    # Add to appropriate queue
                    if token == EOS or new_length >= max_out_len:
                        beam_search.add_final(score=new_node.eval(), node=new_node)
                    else:
                        beam_search.add(score=new_node.eval(), node=new_node)
            
            # Prune beams
            beam_search.prune()
        
        # Get best sequence
        best_score, best_node = beam_search.get_best()
        best_sequence = best_node.sequence.tolist()
        
        # Remove BOS and trim after EOS
        if best_sequence[0] == BOS:
            best_sequence = best_sequence[1:]
        if EOS in best_sequence:
            eos_idx = best_sequence.index(EOS)
            best_sequence = best_sequence[:eos_idx+1]
        
        all_predictions.append(best_sequence)
    
    return all_predictions
