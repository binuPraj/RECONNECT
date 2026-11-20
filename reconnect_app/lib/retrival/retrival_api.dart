import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import 'retrival_config.dart';

class RetrivalApi {
  static Uri _uri(String path) => Uri.parse('${RetrivalConfig.baseUrl}$path');

  static Future<Map<String, dynamic>> _read(http.Response response) async {
    final decoded = response.body.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(response.body) as Map<String, dynamic>;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(
        decoded['detail'] ?? 'Retrieval failed (${response.statusCode})',
      );
    }
    return decoded;
  }

  static Future<String> transcribe({
    required int patientId,
    required File audio,
  }) async {
    final request = http.MultipartRequest('POST', _uri('/transcribe'))
      ..fields['patient_id'] = '$patientId'
      ..files.add(await http.MultipartFile.fromPath('audio', audio.path));
    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);
    final decoded = await _read(response);
    return (decoded['transcript'] as String? ?? '').trim();
  }

  static Future<Map<String, dynamic>> query({
    required int patientId,
    required String question,
    Map<String, dynamic>? faceResult,
  }) async {
    return _read(
      await http.post(
        _uri('/query'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'patient_id': patientId,
          'question': question,
          ...?(faceResult == null ? null : {'face_result': faceResult}),
        }),
      ),
    );
  }

  static Future<Map<String, dynamic>> whoIs({
    required int patientId,
    required String question,
    required File video,
  }) async {
    final request = http.MultipartRequest('POST', _uri('/who-is'))
      ..fields['patient_id'] = '$patientId'
      ..fields['question'] = question
      ..files.add(await http.MultipartFile.fromPath('video', video.path));
    final streamed = await request.send();
    return _read(await http.Response.fromStream(streamed));
  }

  static Future<Map<String, dynamic>> debugLog() async {
    return _read(await http.get(_uri('/debug')));
  }
}
