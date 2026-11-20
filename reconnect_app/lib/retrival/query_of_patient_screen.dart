import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import '../who_is_this_camera_screen.dart';
import 'retrival_api.dart';
import 'retrival_config.dart';

class QueryOfPatientScreen extends StatefulWidget {
  final int patientId;
  final String patientName;

  const QueryOfPatientScreen({
    super.key,
    required this.patientId,
    required this.patientName,
  });

  @override
  State<QueryOfPatientScreen> createState() => _QueryOfPatientScreenState();
}

class _QueryOfPatientScreenState extends State<QueryOfPatientScreen> {
  static const _primary = Color(0xFF4F6F8F);
  static const _text = Color(0xFF2F3B47);

  final _recorder = AudioRecorder();
  final _typed = TextEditingController();
  Timer? _timer;
  int _seconds = 0;
  bool _recording = false;
  bool _busy = false;
  String? _status;
  String? _transcript;
  String? _answer;
  Map<String, dynamic>? _lastResult;

  @override
  void dispose() {
    _timer?.cancel();
    _typed.dispose();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _toggleRecord() async {
    if (_recording) {
      await _stopAndAsk();
      return;
    }
    final allowed = await _recorder.hasPermission();
    if (!allowed) {
      setState(() => _status = 'Microphone permission is required.');
      return;
    }
    final directory = await getTemporaryDirectory();
    final path =
        '${directory.path}/patient_query_${DateTime.now().microsecondsSinceEpoch}.wav';
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.wav),
      path: path,
    );
    setState(() {
      _recording = true;
      _seconds = 0;
      _status = 'Listening… ask your question.';
    });
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() => _seconds++);
      if (_seconds >= 20) {
        _stopAndAsk();
      }
    });
  }

  Future<void> _stopAndAsk() async {
    _timer?.cancel();
    final path = await _recorder.stop();
    setState(() => _recording = false);
    if (path == null) {
      setState(() => _status = 'No audio was captured.');
      return;
    }
    await _askFromAudio(File(path));
  }

  Future<void> _askFromAudio(File audio) async {
    setState(() {
      _busy = true;
      _status = 'Transcribing…';
    });
    try {
      final transcript = await RetrivalApi.transcribe(
        patientId: widget.patientId,
        audio: audio,
      );
      if (transcript.isEmpty) {
        setState(() {
          _busy = false;
          _status = 'I could not hear a question. Please try again.';
        });
        return;
      }
      _typed.text = transcript;
      await _runQuery(transcript);
    } catch (error) {
      setState(() {
        _busy = false;
        _status = '$error';
      });
    }
  }

  Future<void> _askTyped() async {
    final text = _typed.text.trim();
    if (text.isEmpty) {
      setState(() => _status = 'Capture or type a question first.');
      return;
    }
    await _runQuery(text);
  }

  Future<void> _runQuery(String question) async {
    setState(() {
      _busy = true;
      _transcript = question;
      _status = 'Looking that up…';
    });
    try {
      var result = await RetrivalApi.query(
        patientId: widget.patientId,
        question: question,
      );
      if (result['action'] == 'need_camera') {
        result = await _captureWhoIs(question) ?? result;
      }
      if (!mounted) return;
      setState(() {
        _busy = false;
        _lastResult = result;
        _answer = result['answer'] as String? ?? '';
        _status = 'Done.';
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _status = '$error';
      });
    }
  }

  Future<void> _triggerWhoIsThisCamera() async {
    // Passes 'Who is this?' through the parser LLM first before opening the camera
    await _runQuery('Who is this?');
  }

  Future<Map<String, dynamic>?> _captureWhoIs(String question) async {
    setState(() => _status = 'Opening camera for 5-second scan…');
    final video = await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const WhoIsThisCameraScreen()),
    );
    if (video == null) {
      setState(() => _status = 'Camera capture was cancelled.');
      return null;
    }
    setState(() => _status = 'Recognizing person and retrieving memories…');
    return RetrivalApi.whoIs(
      patientId: widget.patientId,
      question: question,
      video: File('${video.path}'),
    );
  }

  @override
  Widget build(BuildContext context) {
    final session = _lastResult?['session'] as Map<String, dynamic>?;
    final resolvedPerson = _lastResult?['resolved']?['person'] as Map<String, dynamic>?;

    return Scaffold(
      backgroundColor: const Color(0xFFF8F6F6),
      appBar: AppBar(
        backgroundColor: const Color(0xFFF8F6F6),
        foregroundColor: _text,
        elevation: 0,
        title: const Text('Query by Patient'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.patientName,
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                        color: _text,
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      'Ask questions or scan someone nearby.',
                      style: TextStyle(color: Colors.black54, fontSize: 13),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Two Primary Action Cards
          Row(
            children: [
              // 1. Microphone Query Button
              Expanded(
                child: InkWell(
                  onTap: _busy ? null : _toggleRecord,
                  borderRadius: BorderRadius.circular(16),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 12),
                    decoration: BoxDecoration(
                      color: _recording
                          ? Colors.redAccent.withValues(alpha: 0.12)
                          : Colors.white,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(
                        color: _recording ? Colors.redAccent : Colors.grey.shade300,
                        width: _recording ? 2 : 1,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.04),
                          blurRadius: 8,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        Container(
                          width: 52,
                          height: 52,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: _recording
                                ? Colors.redAccent
                                : _primary.withValues(alpha: 0.12),
                          ),
                          child: Icon(
                            _recording ? Icons.stop : Icons.mic,
                            size: 28,
                            color: _recording ? Colors.white : _primary,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Text(
                          _recording ? 'Listening ${_seconds}s' : 'Speak Question',
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 14,
                            color: _recording ? Colors.redAccent : _text,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          _recording ? 'Tap to stop' : 'Tap to talk',
                          style: const TextStyle(fontSize: 12, color: Colors.black54),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 14),

              // 2. Who is this Camera Button
              Expanded(
                child: InkWell(
                  onTap: _busy ? null : _triggerWhoIsThisCamera,
                  borderRadius: BorderRadius.circular(16),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 12),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.grey.shade300, width: 1),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.04),
                          blurRadius: 8,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        Container(
                          width: 52,
                          height: 52,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.blue.withValues(alpha: 0.12),
                          ),
                          child: const Icon(
                            Icons.camera_alt_outlined,
                            size: 28,
                            color: Colors.blueAccent,
                          ),
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          'Who is this?',
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 14,
                            color: _text,
                          ),
                        ),
                        const SizedBox(height: 2),
                        const Text(
                          '5s camera scan',
                          style: TextStyle(fontSize: 12, color: Colors.black54),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          // Live Recording / Processing Status Banner
          if (_recording)
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.redAccent.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.redAccent.withValues(alpha: 0.3)),
              ),
              child: Row(
                children: [
                  const SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.redAccent),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Listening… (${_seconds}s) Speak clearly, then tap Speak to stop.',
                      style: const TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Colors.redAccent,
                        fontSize: 13,
                      ),
                    ),
                  ),
                ],
              ),
            )
          else if (_busy && _status != null)
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.blue.shade200),
              ),
              child: Row(
                children: [
                  const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.blueAccent),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      _status!,
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Colors.blue.shade800,
                        fontSize: 13,
                      ),
                    ),
                  ),
                ],
              ),
            ),

          // Primary Output & Answer Card (Immediately Visible!)
          if (_answer != null) ...[
            const SizedBox(height: 16),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.06),
                    blurRadius: 10,
                    offset: const Offset(0, 3),
                  ),
                ],
                border: Border.all(color: Colors.grey.shade200),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: const [
                          Icon(Icons.chat_bubble_outline, size: 20, color: _primary),
                          SizedBox(width: 8),
                          Text(
                            'Answer',
                            style: TextStyle(
                              fontWeight: FontWeight.w800,
                              fontSize: 17,
                              color: _text,
                            ),
                          ),
                        ],
                      ),
                      if (resolvedPerson != null && resolvedPerson['name'] != null)
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.green.shade50,
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(color: Colors.green.shade300),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.check_circle, size: 14, color: Colors.green),
                              const SizedBox(width: 4),
                              Text(
                                '${resolvedPerson['name']}${resolvedPerson['relation'] != null ? ' · ${resolvedPerson['relation']}' : ''}',
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w700,
                                  color: Colors.green.shade800,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                  if (_transcript != null && _transcript!.isNotEmpty) ...[
                    const SizedBox(height: 10),
                    Text(
                      'You asked: "$_transcript"',
                      style: const TextStyle(
                        fontSize: 13,
                        color: Colors.black54,
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                  ],
                  const SizedBox(height: 12),
                  Text(
                    _answer!,
                    style: const TextStyle(
                      fontSize: 17,
                      height: 1.5,
                      fontWeight: FontWeight.w500,
                      color: _text,
                    ),
                  ),
                ],
              ),
            ),
          ],

          const SizedBox(height: 24),
          const Text(
            'Quick Suggestions',
            style: TextStyle(
              fontWeight: FontWeight.w700,
              fontSize: 14,
              color: Colors.black54,
            ),
          ),
          const SizedBox(height: 10),

          // Quick Suggestion Chips
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                _buildPromptChip('Who is Sam?'),
                const SizedBox(width: 8),
                _buildPromptChip('What happened when I met my daughter?'),
                const SizedBox(width: 8),
                _buildPromptChip('What is my schedule today?'),
                const SizedBox(width: 8),
                _buildPromptChip('What is happening right now?'),
              ],
            ),
          ),

          const SizedBox(height: 18),
          TextField(
            controller: _typed,
            minLines: 1,
            maxLines: 3,
            decoration: InputDecoration(
              hintText: 'Or type your question…',
              filled: true,
              fillColor: Colors.white,
              suffixIcon: IconButton(
                icon: const Icon(Icons.send, color: _primary),
                onPressed: _busy ? null : _askTyped,
              ),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide(color: Colors.grey.shade300),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide(color: Colors.grey.shade300),
              ),
            ),
            onSubmitted: (_) => _busy ? null : _askTyped(),
          ),
          if (session != null) ...[
            const SizedBox(height: 20),
            Text(
              'Session ${session['session_id'] ?? ''} · intent ${_lastResult?['intent'] ?? ''} · ${_lastResult?['outcome'] ?? ''}',
              style: const TextStyle(fontSize: 12, color: Colors.black54),
            ),
            Text(
              'Audit parser & answer LLM calls at ${RetrivalConfig.baseUrl}/',
              style: const TextStyle(fontSize: 12, color: Colors.black45),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildPromptChip(String prompt) {
    return ActionChip(
      label: Text(
        prompt,
        style: const TextStyle(fontSize: 13, color: _primary),
      ),
      backgroundColor: Colors.white,
      side: BorderSide(color: Colors.grey.shade300),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      onPressed: _busy
          ? null
          : () {
              _typed.text = prompt;
              _runQuery(prompt);
            },
    );
  }
}
