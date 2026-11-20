import '../app_config.dart';

class RetrivalConfig {
  // Retrieval is hosted by the normal FastAPI backend, not a second server.
  static const int port = 8000;
  static const String path = '/retrieval';

  static String get baseUrl => 'http://${AppConfig.serverHost}:$port$path';
}
