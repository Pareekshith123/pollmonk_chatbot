import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
@Injectable({
  providedIn: 'root'
})
export class MessageService {

  constructor(private http: HttpClient) {}
  sendMessage(message: string) {
    return this.http.post('http://127.0.0.1:5000/chat', { message: message });
  }
}
