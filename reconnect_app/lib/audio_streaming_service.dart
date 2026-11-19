import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

class AudioStreamingService {
  final _recorder = AudioRecorder();
  WebSocketChannel? _channel;
  StreamSubscription<Uint8List>? _micSubscription;
  StreamSubscription? _serverMessages;

  bool _awaitingAck = false;
  final List<Uint8List> _pendingQueue = [];
  bool _isStreaming = false;
  final _events = StreamController<Map<String, dynamic>>.broadcast();

  static const int _sampleRate = 16000;
  static const int _channels = 1;

  Future<void> start(String wsUrl, {int? patientId}) async {
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) throw Exception('Microphone permission denied');

    _channel = WebSocketChannel.connect(Uri.parse(wsUrl));

    // Send metadata handshake first
    _channel!.sink.add(
      jsonEncode({
        'sample_rate': _sampleRate,
        'channels': _channels,
        'encoding': 'pcm_s16le',
        'patient_id': patientId,
      }),
    );

    _serverMessages = _channel!.stream.listen(_handleServerMessage);

    // Start capturing raw PCM16 mono audio
    final stream = await _recorder.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: _sampleRate,
        numChannels: _channels,
      ),
    );

    _isStreaming = true;
    _micSubscription = stream.listen((chunk) {
      if (_awaitingAck) {
        _pendingQueue.add(chunk); // buffer while waiting for previous ack
      } else {
        _sendChunk(chunk);
      }
    });
  }

  void _handleServerMessage(dynamic message) {
    final data = jsonDecode(message as String);

    if (data['event'] == 'stream_closed') {
      _events.add({'event': 'stream_closed'});
      return;
    }
    if (data['event'] == 'capture_queue_ready') {
      _events.add({'event': 'capture_queue_ready'});
      return;
    }

    if (data['received'] != true) {
      // Server rejected something — stop to avoid a silent broken stream
      stop();
      throw Exception(data['error'] ?? 'Server rejected audio stream');
    }

    final identityResult = data['identity_result'];
    if (identityResult is Map) {
      _events.add(Map<String, dynamic>.from(identityResult));
    }

    // Processing events are delivered independently of audio acknowledgements.
    // They must not release the next microphone chunk before its own reply.
    final isAudioAcknowledgement = data.containsKey('chunk_number');
    if (!isAudioAcknowledgement) return;

    _awaitingAck = false;

    // If more chunks arrived while we were waiting, send the next one now
    if (_pendingQueue.isNotEmpty) {
      final next = _pendingQueue.removeAt(0);
      _sendChunk(next);
    }
  }

  void _sendChunk(Uint8List chunk) {
    _awaitingAck = true;
    _channel!.sink.add(chunk);
  }

  Future<void> stop() async {
    _isStreaming = false;
    await _micSubscription?.cancel();
    await _recorder.stop();
    _pendingQueue.clear();
    _awaitingAck = false;

    if (_channel != null) {
      try {
        _channel!.sink.add(jsonEncode({'action': 'flush_and_stop'}));
      } catch (e) {
        debugPrint('[STREAM] Error sending flush_and_stop: $e');
        await disconnect();
      }
    }
  }

  Future<void> disconnect() async {
    await _serverMessages?.cancel();
    await _channel?.sink.close();
    _channel = null;
  }

  /// Tell the backend that every queued unknown-voice camera capture has
  /// finished, so it may safely close this microphone session.
  Future<void> completeCaptureQueue() async {
    if (_channel == null) return;
    try {
      _channel!.sink.add(jsonEncode({'action': 'capture_queue_complete'}));
    } catch (e) {
      debugPrint('[STREAM] Error acknowledging completed capture queue: $e');
    }
  }

  bool get isStreaming => _isStreaming;

  Stream<Map<String, dynamic>> get events => _events.stream;

  Future<void> dispose() async {
    await stop();
    await disconnect();
    await _recorder.dispose();
    await _events.close();
  }
}
