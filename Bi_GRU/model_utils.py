import tensorflow as tf
import model

def create_model(session, encoder_size, decoder_size, hidden_dim, input_dim, output_dim, load_model=0, checkpoint_dir=""):
    model = seq2seq_model.Seq2SeqModel(
        encoder_size=encoder_size,
        decoder_size=decoder_size,
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        output_dim=output_dim)

    print "load_model", load_model == 1
    if load_model == 1:
        model.saver.restore(session, checkpoint_dir)
    else:
        session.run(tf.global_variables_initializer())
    return model